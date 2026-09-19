"""Common scoring window for runs of DIFFERENT billet length.

WHY THIS EXISTS. Two reported defects turned out to be one defect seen twice:

  metrics -- a whole-bar statistic over a 92 mm bar and a 59 mm bar is not the same
             measurement. `detF_p01` is a PERCENTILE, so a longer bar's extra pristine
             particles dilute it UPWARD; `detF_min` is an EXTREME VALUE, so more draws
             push it DOWN. Biased in OPPOSITE directions, so neither is sample-size
             neutral and an effect that moves only one of them is suspect.
  renders -- [Thomas, 2026-09-19] "a thin region of red particles on the L 92 and L 125
             runs", and the error renders "don't adjust/account for the increased
             length". Same cause: the longer bar carries material with no counterpart in
             the reference, and every aggregate silently includes it.

THE REGISTRATION, which decides everything below. `Hit.z` is measured from the PINNED
FACE (`forge_common.hit`); the adapter sets `qpos[0,0] = x_pinned_face - z`; and
`batch_arms` shifts every hit z by `+dL` when the billet is lengthened. That shift holds
each strike a FIXED distance from the FREE end while the wall retreats. So:

    the forged features are FREE-END registered, and the added metal is ALL at the
    pinned end.

Verified, not assumed: the 17-hit program reads z 60.48..82.53 mm at L=92 and
27.48..49.53 mm at L=59, which is the same u = L - z to the millimetre.

GEOMETRY (agforge/options.py). `cylinder_pos` is x=0, so the billet spans [-L/2, +L/2];
`fixed_region_center = cylinder_pos + [0.5*L, 0, 0]` and `fixed_region_size_x = 0.35*L`,
so the clamp occupies x in [0.325*L, 0.675*L] and its INNER EDGE sits at z = 0.175*L
from the pinned face.

THE TWO EXCLUSIONS.
  * reference extent -- material past the real stock's own length has nothing to be
    scored against. At L=92 that is exactly the 33 mm synthesized stub. A correctly
    clipped score is therefore robust to the stub's geometry by construction, which
    matters because that stub is (2026-09-19) known to be defective.
  * clamp artifact -- the fixed-end BC corrupts a band just inside the clamp. Measured
    at ~<=1.2*dx, consistent with a P2G stencil half-width. Excluded as
    0.175*L + CLAMP_ARTIFACT_DX * dx, measured from the PINNED face, so it differs per
    run and must be intersected across the set being compared.

The free end needs no exclusion -- it is real material in every run.
"""
import numpy as np

# ------------------------------------------------------------------ "how long is the real bar?"
# Four different numbers were in use for this on 2026-09-19, a 1.1% spread, each hardcoded
# somewhere different. They are NOT all the same quantity, so pinning one value would itself be
# wrong; what was missing is a statement of which means what. Import from here, do not re-literal.
#
#   59.0     NOMINAL, and the REGISTRATION DATUM. forge_common.real_scale.REAL_STOCK_LENGTH_MM.
#            batch_arms computes dL = stock_length - 59.0 and shifts every hit z by it, so this
#            value DEFINES the shift. It must not be "corrected" to the measured extent: doing so
#            would silently re-register every run ever recorded. Authoritative for dL only.
#   59.196   MEASURED extent of the scan mesh (billet_hit01_before_d8000.obj, bbox in x).
#            Authoritative for anything about the real billet's actual geometry -- volume per mm,
#            what a synthesized extension has to match, whether a rebuilt mesh is the right size.
#   58.98    the real part's free end in canonical x AT HIT 1. Not a constant, and not a competing
#            definition: the part elongates under forging, so the same measurement reads 71.01 at
#            hit 8 and 93.08 at hit 17. Read it per hit from real_meshes/hit_NN.npz.
#   58.57    ORPHAN. Appeared only as a bare literal in score_arms.py, with no derivation anywhere
#            in the tree. Removed 2026-09-19. Do not reintroduce.
REAL_STOCK_LENGTH_MM = 59.0        # nominal; DEFINES the hit-z shift. Read the note before changing.
SCAN_STOCK_LENGTH_MM = 59.196      # measured bbox extent of the scan mesh
REAL_STOCK_RADIUS_MM = 20.0        # forge_common.real_scale; note this is the BOUNDING BOX radius,
                                   # ~10.9% more material than the scan actually holds

CLAMP_ARTIFACT_DX = 1.2       # measured band width, in dx. See BACKLOG "clamp boundary artifact".

# THE CLAMP IS PROPORTIONAL, NOT ABSOLUTE (options.py: fixed_region_size = 0.35*cylinder_height),
# so lengthening the bar lengthens the CLAMP as well as the clearance. The 2026-09-19 length
# result changed both at once:
#           clamp length   inner edge   nearest strike   clearance
#     L=59     20.65 mm      10.33 mm      23.43 mm       13.1 mm
#     L=92     32.20 mm      16.10 mm      56.43 mm       40.3 mm
# Clearance x3.1 and clamp x1.56 in one step, so "length is what fixes the clamp" attributes to
# clearance an effect that clamp length is an uncontrolled co-variate in. Raised by
# free-cloud-compute-branch-5, verified here at options.py:288. Separating them is one run:
# L=92 with AGF_CLAMP_FRACTION set so the clamp holds the same ABSOLUTE 20.65 mm as L=59.
CLAMP_FRACTION_OF_LENGTH = 0.35


def to_free_end_mm(x_mm, stock_length_mm):
    """Raw-sim x (mm) -> distance from the FREE end (mm). u=0 free end, u=L pinned face."""
    return np.asarray(x_mm, dtype=np.float64) + 0.5 * float(stock_length_mm)


def to_pinned_face_mm(x_mm, stock_length_mm):
    """Raw-sim x (mm) -> distance from the PINNED face (mm), the `Hit.z` convention."""
    return 0.5 * float(stock_length_mm) - np.asarray(x_mm, dtype=np.float64)


def clean_u_max(stock_length_mm, dx_mm, real_length_mm=REAL_STOCK_LENGTH_MM):
    """Largest u this ONE run can be scored to: the nearer of the reference extent and
    the clamp artifact's outer reach."""
    L = float(stock_length_mm)
    by_reference = min(L, float(real_length_mm))
    by_clamp = 0.825 * L - CLAMP_ARTIFACT_DX * float(dx_mm)
    return min(by_reference, by_clamp)


def common_u_max(runs):
    """Intersection over runs. `runs` is an iterable of (stock_length_mm, dx_mm)."""
    return min(clean_u_max(L, dx) for L, dx in runs)


def mask(pos_mm, stock_length_mm, u_max, u_min=0.0):
    """Boolean mask selecting particles inside [u_min, u_max] from the free end.

    `pos_mm` is (N,3) raw-sim position in MILLIMETRES."""
    u = to_free_end_mm(np.asarray(pos_mm)[:, 0], stock_length_mm)
    return (u >= float(u_min)) & (u <= float(u_max))


def describe(stock_length_mm, dx_mm, real_length_mm=REAL_STOCK_LENGTH_MM):
    L = float(stock_length_mm)
    return {
        "stock_length_mm": L,
        "dx_mm": float(dx_mm),
        "clamp_inner_edge_z_mm": 0.175 * L,
        "clamp_artifact_reach_z_mm": 0.175 * L + CLAMP_ARTIFACT_DX * float(dx_mm),
        "reference_extent_u_mm": min(L, float(real_length_mm)),
        "clean_u_max_mm": clean_u_max(L, dx_mm, real_length_mm),
    }


# ------------------------------------------------------------------ canonical-frame registration
# The stored `*_hits.npz` clouds are in the CANONICAL mm frame the adapter writes:
#     canon_x = (x_pinned_face_m - sim_x) * 1000
# so canon_x = 0 at the PINNED face and increases toward the free end. The real scan meshes
# (`real_meshes/hit_NN.npz`) use the same frame and the same origin.
#
# 🚨 THAT SHARED ORIGIN IS A TRAP FOR A LENGTHENED RUN. `batch_arms` shifts every commanded hit
# z by +dL, so the forged features sit dL further from the pinned face than the real part's do.
# Overlay the two on canon_x as stored and a 92 mm run is compared against the real part shifted
# by 33 mm -- the sim's forged zone against the real part's unforged shank. With the renderer's
# +-3 mm colour clamp, everything saturates.
#
# MEASURED, 2026-09-19, mesh92_n17 vs real_meshes:
#     raw    free end 91.99 vs real 58.98   -> off +33.01 mm
#     minus dL                              -> off  +0.01 mm
#     commanded hit z 56.43..94.60, minus dL -> 23.43..61.60, identical to the L=59 run
# So subtracting `hit_z_shift_mm` registers the FREE ENDS, which is the registration the hit
# program already uses. Residual offsets at later hits (+1.76 / -2.81 mm at hits 8 / 17) are
# genuine sim-vs-real forging differences and are what the comparison is for.


def align_to_reference(canon_x_mm, hit_z_shift_mm):
    """Canonical x as stored -> canonical x registered against the real scan (free ends)."""
    return np.asarray(canon_x_mm, dtype=np.float64) - float(hit_z_shift_mm)


def scored_window_canon(stock_length_mm, dx_mm, hit_z_shift_mm, real_free_end_mm=None):
    """[lo, hi] in ALIGNED canonical mm holding material that can honestly be scored.

    The LOWER bound is the real exclusion and it is asymmetric on purpose. Below it sits
    material past the real part's own pinned face -- the synthesized stub on a lengthened bar --
    and material the fixed-end BC has corrupted. Neither is error; colouring it red is what
    produced the reported "thin region of red particles on the L 92 and L 125 runs".

    The UPPER bound defaults to none, and that is deliberate. Sim material beyond the real
    part's free end is NOT unmatched -- it is genuine OVER-ELONGATION, which is exactly the
    error a signed-distance render exists to show. Clipping there would hide a real defect.
    Pass `real_free_end_mm` only for an AGGREGATE over a matched extent, where material with no
    counterpart to difference against would bias the statistic."""
    lo = max(0.0, 0.175 * float(stock_length_mm)
             + CLAMP_ARTIFACT_DX * float(dx_mm) - float(hit_z_shift_mm))
    hi = float("inf") if real_free_end_mm is None else float(real_free_end_mm)
    return lo, hi


def scored_mask_canon(canon_x_mm_aligned, stock_length_mm, dx_mm, hit_z_shift_mm,
                      real_free_end_mm=None):
    lo, hi = scored_window_canon(stock_length_mm, dx_mm, hit_z_shift_mm, real_free_end_mm)
    x = np.asarray(canon_x_mm_aligned, dtype=np.float64)
    return (x >= lo) & (x <= hi)
