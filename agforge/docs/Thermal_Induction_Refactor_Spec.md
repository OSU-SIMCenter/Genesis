# Thermal / Induction Heating Refactor — Implementation Spec

**Status:** Draft for review · **Scope:** `agforge/thermal.py`, `agforge/strike_controller.py`,
`agforge/options.py`, `genesis/engine/solvers/base_mpm_solver.py`,
`genesis/engine/couplers/legacy_coupler.py`, `genesis/options/solvers.py`
**Author:** (drafted by Claude for Tim) · **Date:** 2026-06-07

---

## 0. Goal

Make the induction heating and fixed-end boundary condition physically accurate, fast
(GPU-resident, no per-frame CPU round-trips), reasonably differentiable, and idiomatic to
the Genesis engine — i.e. they should "blend in" with the existing thermal kernels
(`p2g_post_constitutive`, `mpm_grid_op`, `mpm_grid_thermal_diffusion`) rather than sit on
top as Python add-ons.

Three deliverables, phased so the sim is always in a working state:

- **Phase 0** — revert the broken staged change; restore a correct *behaving* baseline.
- **Phase 1** — the real refactor: induction as a precomputed-SDF + in-kernel volumetric
  particle source; fixed-end Robin BC (constant-ambient); toggleable visual fade.
- **Phase 2** — later sophistication: 1-D axial bulk bar, enthalpy-form conduction,
  edge-effect flux crowding, post-deformation grid-surface distance.

> **Note on differentiability:** treated as a *nice-to-have*, not a hard requirement. The
> design keeps gradients flowing where it's free to do so (the induction source is a plain
> particle-temp update on `needs_grad` fields), but we will **not** invest in validating or
> testing gradients, and we will not sacrifice performance or clarity to preserve them.

---

## 1. Design principles

**1.1 Physics taxonomy → numerical location.** Heat effects are placed where they belong:

| Effect | Nature | Lives on | Existing example to mirror |
|---|---|---|---|
| Adiabatic plastic heating | volumetric body source | **particles** (in P2G) | `p2g_post_constitutive` |
| **Induction heating** | volumetric body source | **particles** (in P2G) | ← new, sibling of the above |
| Conduction / diffusion | internal transport | grid | `mpm_grid_thermal_diffusion` |
| Air convection + radiation | surface flux | grid surface cells | `mpm_grid_op` |
| Die contact cooling | surface flux | particles near die SDF | `apply_particle_thermal_contact` |
| **Fixed-end bulk conduction** | surface flux (cut plane) | **grid cut-face cells** | ← new, sibling of air convection |

Induction is a *body source* attached to material, so it goes on particles next to
adiabatic heating. The fixed-end BC is a *surface flux* on the truncation plane, so it goes
on the grid next to convection — but only after the cut-face cells are correctly
distinguished from air-exposed surface cells.

**1.2 Precompute vs per-frame split (induction).** The heat intensity factors into:

```
dT_i  =  q_peak · w_i · f_axial(x_i ; coil) · S_T · substep_dt / (ρ · Cp(T_i))
              └──┘   └──────────────────┘
         precomputed     per-frame analytic
         (SDF depth)     (coil slides each frame)
```

- `w_i = exp(-2·d_i/δ)` — skin-depth weight from the SDF depth `d_i`. **Geometry-only and
  expensive → compute once per strike (and at init / on restore), store on the GPU.**
- `f_axial` — normalized Biot–Savart axial profile; depends on the *moving* coil position →
  cheap analytic, evaluated **per substep in-kernel**.

**1.3 Conventions.** New kernels are `@qd.func`/`@qd.kernel`, guarded by
`qd.static(self._enable_thermal)`, scaled by `self._thermal_time_scale` exactly like the
existing cooling/diffusion, use `mass_thermal_real = mass_thermal / _particle_volume_scale`
for energy math, accumulate a `dT_*` telemetry field (matching `dT_conv`/`dT_rad`), and read
mutable per-frame scalars from a `qd.field` (the `_gravity` pattern), never from baked Python
floats.

**1.4 Physics stays pure; visuals are a separate channel.** No physics quantity is ever
clamped for the renderer. The Unity color fade operates on a throwaway copy and is
toggleable.

---

## 2. Phase 0 — baseline cleanup (no engine changes)

The working tree currently has a **staged** change to `thermal.py` that re-introduces
billet-volume scaling (the approach we rejected) and has a latent `NameError`
(`B_sq`/`dx` referenced outside the `coil_center is not None` block).

**Action:**
1. `git restore --staged --worktree agforge/thermal.py` — drop the staged volume math,
   returning to the committed `0c7a650e` version.
2. In the committed version, make two small corrections so the *current* Python path behaves
   correctly while Phase 1 is built:
   - Normalize the axial profile to its own peak: `f = B_sq / B_peak_sq`, with
     `B_peak_sq = (h / sqrt(h² + R²))²` (closed form for the centered finite solenoid).
   - Re-interpret `heating_power` as a **peak volumetric power density `q_peak` [W/m³]**
     (delete the `q_max = surface_power / volume_coil` line; use `q_peak` directly) and bump
     it to a value that beats cooling (see §6 / calculator; ballpark `1e8–1e9 W/m³`).

This makes the existing Python heater correct-behaving and gives a reference to validate
Phase 1 against. Phase 0 is throwaway — Phase 1 deletes this path entirely.

---

## 3. Phase 1a — Induction: precomputed SDF + in-kernel volumetric source

### 3.1 New per-particle depth field (standalone, not double-buffered)

Add a standalone solver field for the precomputed skin depth — **not** part of the
double-buffered `particle_state` struct (it is static between strikes, needs no gradient, and
we don't want it in `copy_frame_helper` or the per-substep frame copy):

```python
# base_mpm_solver.py, in build()/init, alongside other fields
self.induction_depth = qd.field(dtype=gs.qd_float, shape=(self._n_particles, self._B))
self.induction_depth.fill(1.0e9)   # default: "infinitely deep" => zero heating
```

Rationale for storing **depth** (not the `exp` weight): keeps `skin_depth (δ)` a live-tunable
parameter, and the `exp` is trivial on the GPU.

Rationale for **particles, not grid nodes**: temperature/plastic state already live on
particles; a particle field is material-attached (survives the per-substep grid reset, moves
with the metal), is finer than the grid, and is consistent with the adiabatic-heating term we
are mirroring.

### 3.2 SDF recompute hook (once per strike)

The SDF is computed on the CPU via `igl.signed_distance` (kept — it handles arbitrary forged
geometry, which the analytic radial proxy cannot) and the result uploaded to
`induction_depth` once per geometry change.

**When to recompute** (all currently available signals):
- at **init** (billet is the initial cylinder);
- on **strike completion** — the `RELEASE → IDLE` transition in
  `strike_controller.update_logic` (≈ line 481), which is exactly "after each strike";
- on **checkpoint restore / undo** — positions change, so depth must be recomputed
  (or saved+restored; recompute is simpler and authoritative).

Keep keying off `reconstructor.mesh_version` as the dedup signal (it already bumps on
reconstruction and could bump on restore), but gate the *expensive* path so it only fires on
those events, never every frame.

**New API on the solver/entity** (mirrors `set_gravity` semantics):

```python
def set_induction_depth_from_mesh(self, verts, faces):
    """Recompute per-particle skin depth from a surface mesh (CPU igl), upload to GPU.
       Call at init, after each strike, and on checkpoint restore."""
    pos = self.get_particles_pos().cpu().numpy().reshape(-1, 3)
    d, _, _, _ = igl.signed_distance(pos.astype(np.float64), verts, faces)
    depth = np.abs(d)
    depth[~np.isfinite(depth)] = 1.0e9          # NaN/inf => no heating (current guard)
    self._kernel_upload_induction_depth(depth)  # host->device, once per strike
```

(The mesh-acquisition / degenerate-mesh guards from the current `step_heat` move here
verbatim.)

### 3.3 Coil parameter field (mutable per-frame uniforms — the `_gravity` pattern)

```python
# base_mpm_solver.py init
self._induction = qd.field(dtype=induction_params_struct, shape=())
# fields: center (vec3), half_length, radius, q_peak, skin_depth, active (i32)
```

Python sets these **once per macro-step**, before `scene.step()`, from the controller:

```python
def set_induction_params(self, center, half_length, radius, q_peak, skin_depth, active):
    ...  # write into self._induction[None]
```

The controller computes `center` from the sliding-arm position
(`current_slider_x + coil_offset_x`, `y=0`, `z=cylinder_pos[2]`) and sets `active=1` only
when not striking (and the coil is "powered"), `0` otherwise. Within the 8 substeps the
values are constant; the kernel reads them each substep.

### 3.4 The induction `@qd.func` (in P2G, sibling of adiabatic heating)

```python
@qd.func
def p2g_induction(self, f, i_p, i_b):
    if qd.static(self._enable_thermal):
        if self._induction[None].active == 1:
            p = self.particles[f + 1, i_p, i_b].pos        # post-constitutive frame
            rho = self.particles_info[i_p].mass / qd.math.max(self._particle_volume, gs.EPS)
            Cp  = self.get_steel_cp(self.particles[f + 1, i_p, i_b].temp)

            # skin-depth weight (precomputed depth, live delta)
            d  = self.induction_depth[i_p, i_b]
            w  = qd.math.exp(-2.0 * d / self._induction[None].skin_depth)

            # normalized finite-solenoid axial profile B(x)^2 / B(0)^2
            x  = p[0] - self._induction[None].center[0]
            h  = self._induction[None].half_length
            R  = self._induction[None].radius
            t1 = (x + h) / qd.math.sqrt((x + h) * (x + h) + R * R)
            t2 = (x - h) / qd.math.sqrt((x - h) * (x - h) + R * R)
            B  = 0.5 * (t1 - t2)
            Bp = h / qd.math.sqrt(h * h + R * R)            # B(0), peak
            f_axial = (B * B) / qd.math.max(Bp * Bp, gs.EPS)

            q_eff = self._induction[None].q_peak * self._thermal_time_scale
            dT = q_eff * w * f_axial * self.substep_dt / (rho * Cp)

            self.particles[f + 1, i_p, i_b].temp += dT
            self.particles[f + 1, i_p, i_b].dT_induction += dT   # telemetry
```

Call it in `p2g` immediately after `p2g_post_constitutive(...)` (≈ line 504), inside the
`active` particle loop. This means induction heat is deposited, then **diffused and cooled in
the same substep** — no operator-split, no Python round-trip.

**Time-scaling consistency:** running 8 substeps × `substep_dt` × `S_T` integrates to
`q_peak · S_T · macro_dt` per macro-step = `q_peak · thermal_dt` — identical total to the old
once-per-frame formulation, but now correctly interleaved with cooling.

### 3.5 New telemetry / bookkeeping fields

- Add `dT_induction: gs.qd_float` to the particle state template (next to
  `dT_adiabatic`/`dT_conv`/`dT_rad`), and to `g2p_prologue` carry-over and
  `copy_frame_helper` (one line each), matching the existing pattern.

### 3.6 Removal of the old path

- Delete the Python `InductionHeater.step_heat` per-frame invocation in
  `strike_controller.step_simulation` (≈ lines 605-631) and the surrounding GPU↔CPU temp
  snapshots used only for it.
- `agforge/thermal.py`: keep `get_steel_cp_numpy` (used elsewhere); the `InductionHeater`
  class is reduced to (or replaced by) the CPU SDF helper feeding
  `set_induction_depth_from_mesh`. The Biot–Savart math now lives in the kernel.

### 3.7 What this buys

- **Performance:** zero per-frame GPU↔CPU syncs for heating; the only CPU cost (igl SDF +
  one upload) happens once per strike.
- **Differentiability:** the source is a plain `temp +=` on `needs_grad` particle fields;
  gradients flow through `q_peak`, coil `center`, `skin_depth`, and `Cp(T)` for free, with
  `induction_depth` as a frozen constant (like `mass`). The discrete SDF recompute is not
  differentiated — correct and expected.
- **Volumetric:** `exp(-2d/δ)` with `δ = R_billet/3` deposits through the skin volume, not
  just the surface.

---

## 4. Phase 1b — Fixed-end Robin BC (constant ambient, Level 1)

### 4.1 The problem with today's code

`mpm_grid_op` flags any metal cell touching an empty cell as `is_surface` and applies air
convection + radiation. The fixed-end **cut plane** (metal whose +X neighbor is empty) is
therefore mis-treated as *air-exposed*, when physically it is welded to the unsimulated rod.
On top of that, `_apply_fixed_end_heat_sink` (Python) applies an aggressive, **unscaled**
lerp toward ⅓ of the active average — non-physical and framerate-dependent.

### 4.2 Cut-face reclassification

Replace the binary `is_surface` flag in `mpm_grid_op` with a 3-way classification per cell:

- `INTERIOR` — no empty neighbors.
- `AIR_SURFACE` — touches empty **and** is not in the fixed-end cut region → convection + radiation (unchanged).
- `CUT_FACE` — metal whose +X neighbor is empty **and** cell-center `x ≥ x_cut − dx`,
  where `x_cut = cylinder_center_x + cylinder_height/2` → **bulk-conduction Robin BC, and
  skip air/radiation** (it is not air-exposed).

`x_cut` is passed in via the induction/coil param field family (it's a known static
geometry value; can live in a small `_thermal_bc` uniform field or be derived from solver
config).

### 4.3 The Robin flux (mirror the air-convection block exactly)

For `CUT_FACE` cells, drive temperature toward `T_bulk` (= ambient in Level 1) with a bulk
conductance, using the same stable exponential update as air convection:

```
h_bulk = k(T_cell) / L_eff                  # conduction into the continuing rod
k_bulk = (h_bulk · A_cut) / (mass_thermal_real · Cp) · S_T
dT_bulk = (T_cell - T_bulk) · (1 - exp(-k_bulk · substep_dt))
grid.temp -= dT_bulk
grid.dT_bulk -= dT_bulk                      # telemetry
```

- `A_cut = dx²` (the cut-plane face area per cell).
- `L_eff` is the one tunable physical parameter (effective conduction length into the bulk);
  calibrate to `√(π·α·t_char)` for a representative heating time (see calculator).
- `k(T)` from the existing `get_steel_thermal_conductivity`.
- Scaled by `S_T = _thermal_time_scale`, exactly like `h_air`.

### 4.4 Removal / config

- Delete `_apply_fixed_end_heat_sink` and its call (`strike_controller.py` ≈ 148-205, 665).
- New options in `genesis/options/solvers.py` (`MPMOptions`, next to the other thermal
  coeffs):
  - `fixed_end_conduction_length: float` (`L_eff`, metres) — default e.g. `0.05`.
  - `fixed_end_ambient: float` — default `293.15`.
- Plumb `x_cut` from the agforge layer (it knows the billet geometry).

> The unsimulated rod warming up over time (the rod is **not** an infinite ambient sink) is
> deliberately deferred to Phase 2 (§6). Level 1 is the simple, stable, correct-in-form
> version.

---

## 5. Phase 1c — Visual fade toggle

The fade already exists and is already display-only
(`update_and_get_recon_data`, `strike_controller.py:969-983`): it blends the 11–21% clamp
zone down to `min(T, 900K)` on a CPU copy of `particles_temp` before the KD-tree maps temps
to mesh vertices. Physics is untouched. Two changes:

1. **Toggle.** Add `EnvOptions.thermal_visual_fade: bool = True` (or under a viz/teleop
   config). Wrap the fade block (981-983) in `if cfg...thermal_visual_fade:`. When off, raw
   physical temperatures are sent to Unity for debugging.
2. **Smooth + parameterize.** Optionally replace the hard `min(T, 900)` + linear `alpha`
   with a `smoothstep` toward the Draper point (~798K) over a configurable fraction, so the
   seam blends rather than clips. Keep the *static-coordinate* evaluation
   (`mapping_parts_np[:,0]`) so the handle is identified regardless of teleop translation.

No new render channel is required — the existing throwaway-copy approach is already the
correct architecture; we are only gating and smoothing it.

---

## 6. Phase 2 — deferred sophistication (not implemented now)

- **1-D axial bulk bar** (replaces Level-1 ambient): a small 1-D heat-equation array
  representing the unsimulated rod beyond the cut plane — own lateral air cooling, far-end
  chuck BC, coupled at the cut plane by flux continuity. Captures the rod heating up and the
  gradient relaxing over a session. Cheap, stateful (checkpointable), differentiable.
  The Robin BC then drives toward the bar's near-end temperature instead of fixed ambient.
- **Enthalpy-form conduction:** diffuse `∫ρCp dT` (or flux form `∇·(k∇T)`) instead of `T`,
  for exact energy conservation across the 450→750 J/kgK `Cp` swing.
- **Edge-effect flux crowding:** amplify induction near high-curvature regions
  (Thermal_Physics_Roadmap §1).
- **Post-deformation skin depth:** on-grid surface-distance field (no CPU igl) for induction
  accuracy on heavily forged geometry.

---

## 7. Parameters & where they live

| Param | Symbol | Location | Default | Units |
|---|---|---|---|---|
| Peak induction power density | `q_peak` | `agforge/options.py` `heating_power` (re-typed) | TBD via calculator (`~1e8–1e9`) | W/m³ |
| Skin depth | `δ` | `agforge/options.py` `skin_depth` | `R_billet/3` | m |
| Coil half-length | `h` | `RobotOptions.coil_length/2` | `0.0185` | m |
| Coil radius | `R` | `RobotOptions.coil_radius` | `0.04` | m |
| Bulk conduction length | `L_eff` | `MPMOptions.fixed_end_conduction_length` | `0.05` | m |
| Fixed-end ambient | `T_bulk` | `MPMOptions.fixed_end_ambient` | `293.15` | K |
| Cut-plane x | `x_cut` | derived: `cyl_center_x + cyl_height/2` | — | m |
| Visual fade enable | — | `EnvOptions.thermal_visual_fade` | `True` | bool |

All physical thermal coefficients continue to be scaled by `thermal_time_scale` (derived
from the thermal CFL limit in `options.py:model_post_init`).

---

## 8. Files touched (Phase 1 checklist)

- `genesis/engine/solvers/base_mpm_solver.py`
  - particle state: `+ dT_induction`
  - fields: `+ induction_depth`, `+ _induction` (params struct), `+ _thermal_bc` (x_cut)
  - kernels: `+ p2g_induction` (called in `p2g`), `+ _kernel_upload_induction_depth`
  - API: `+ set_induction_params`, `+ set_induction_depth_from_mesh`
  - `g2p_prologue` / `copy_frame_helper`: carry `dT_induction`
- `genesis/engine/couplers/legacy_coupler.py`
  - `mpm_grid_op`: 3-way cell classification + Robin `CUT_FACE` branch; skip air/rad on cut face
- `genesis/options/solvers.py`
  - `MPMOptions`: `+ fixed_end_conduction_length`, `+ fixed_end_ambient`
- `agforge/options.py`
  - `heating_power` re-documented as W/m³ peak density; `+ thermal_visual_fade`
- `agforge/strike_controller.py`
  - remove `step_heat` per-frame path + temp snapshots
  - remove `_apply_fixed_end_heat_sink`
  - call `set_induction_params` each macro-step; call `set_induction_depth_from_mesh` at
    init / strike-end (RELEASE→IDLE) / restore
  - gate the visual-fade block on the new toggle
- `agforge/thermal.py`
  - keep `get_steel_cp_numpy`; reduce `InductionHeater` to the CPU SDF→depth helper

---

## 9. Validation plan (qualitative — no gradient testing)

1. **Heating magnitude:** with `q_peak` from the calculator, a fully-immersed billet should
   reach forging temps (~1300K surface) over a realistic dwell, and net heating must exceed
   surface cooling at the target temperature (the failure mode of the committed version).
2. **Geometry independence:** a partially-inserted tip and a fully-immersed billet should
   show the **same** surface heating *rate* under the coil (no "magic laser" tip explosion,
   no volume-coupling weakness).
3. **Axial profile:** visible smooth bell-shaped glow centered on the coil, fading past its
   ends — not a hard band.
4. **Fixed end:** smooth cool gradient toward the held end, framerate-independent (vary FPS /
   step count — gradient should not change, unlike the old unscaled lerp).
5. **Visual toggle:** off → raw physics colors (debug); on → seam fades to dark near the
   handle with no physics change (verify by reading raw temps with toggle off).
6. **Perf:** confirm the per-frame `teleop_heating` profile cost drops to ~0 (no CPU sync);
   confirm igl SDF cost appears only at init/strike-end.

---

## 10. Risks / open questions

- **`q_peak` calibration** is the main unknown → resolve with the companion energy-balance
  calculator (pick `q_peak`, `L_eff`, `h_air`, `ε` to hit a target K/s and steady-state
  gradient *before* running the full sim).
- **Reading post-constitutive frame** `f+1` in `p2g_induction`: confirm the field write
  ordering matches `p2g_post_constitutive` (which already writes `temp` at `f+1`). Apply
  induction *after* it so they compose additively.
- **Checkpoint/restore of `induction_depth`:** simplest correct behavior is to recompute on
  restore (positions changed); confirm the restore path can trigger
  `set_induction_depth_from_mesh`.
- **`_induction` uniform field on a `@qd.data_oriented` solver:** verify Quadrants allows a
  scalar struct `qd.field(shape=())` read in-kernel (the `_gravity` FIXME notes Ndarray
  attrs aren't supported on data-oriented classes in kernel scope — a `qd.field` is the
  supported workaround, which is what we use).
- **CPIC separation vs. induction:** induction is purely a particle-local `temp +=`, so it is
  independent of CPIC normals; no interaction expected.

---

## 11. Suggested sequencing

1. Phase 0 (minutes) — revert + correct committed path; sanity-run.
2. Calculator — pin `q_peak`, `L_eff`.
3. Phase 1a — induction field + kernel + hooks; validate magnitude & geometry-independence.
4. Phase 1b — Robin BC + cut-face reclassification; validate gradient/framerate independence.
5. Phase 1c — visual toggle.
6. Phase 2 — as needed.
