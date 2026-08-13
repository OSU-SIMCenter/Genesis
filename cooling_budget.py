#!/usr/bin/env python3
"""Can free-surface losses account for the measured within-bout cooling rate?

Section 3.4 measures a median -6.03 C/s between blows inside a forging bout. That
is now the only real thermal ground truth this project has, and it is the number
a coupled model has to reproduce. Before running any coupled model it is worth
knowing what the sim's OWN boundary parameters predict, because if free-surface
loss alone cannot reach the measured rate then contact conduction dominates and
the calibration effort belongs there rather than on emissivity.

Everything here is a lumped, uniform-temperature estimate for a bare cylinder. It
is a BOUND, not a simulation: it deliberately ignores internal gradients (which
slow surface-limited cooling further, making the gap worse, not better).
"""

SIGMA = 5.670374419e-8

# --- sim's own shipped parameters (agforge/options.py) ---
EPS_SIM = 0.40          # options.py:705
H_AIR = 15.0            # W/(m^2 K), convection to still air
T_AMB = 293.0           # K
K_CONTACT = 3000.0      # W/(m^2 K), thermal_contact_conductivity, options.py:693

# --- geometry: 38.1 mm round bar (forge-mcap-ground-truth) ---
R = 0.0381 / 2.0
RHO = 7334.0            # kg/m^3, options.py

# --- measured (section 3.4) ---
MEASURED = 6.03         # C/s, median within-bout


def cp_316l(T):
    """Mirrors base_mpm_solver.get_steel_cp / material_properties.cp_316l."""
    if T >= 1000.0:
        return 605.2 + (T - 1000.0) * 0.0743
    return 500.0  # only the >=1000 K branch matters at forging heat


def rate(T_c, eps, h, contact_fraction=0.0):
    """Cooling rate in C/s for a lumped cylinder at T_c degrees Celsius."""
    T = T_c + 273.15
    q_rad = eps * SIGMA * (T ** 4 - T_AMB ** 4)
    q_conv = h * (T - T_AMB)
    q_cond = contact_fraction * K_CONTACT * (T - T_AMB)
    q = q_rad + q_conv + q_cond
    sv = 2.0 / R                      # surface/volume for a long cylinder
    return q * sv / (RHO * cp_316l(T))


print("Within-bout cooling: what do free-surface losses actually buy?")
print(f"  bar 38.1 mm round, rho {RHO:.0f}, Cp(1123 K) {cp_316l(1123.15):.0f} J/kgK")
print(f"  surface/volume = 2/r = {2.0/R:.1f} 1/m")
print()

T_med = 823.6   # session median, section 3.4
print(f"at the session median billet temperature, {T_med} C:")
print(f"{'case':<44}{'C/s':>8}{'vs measured':>13}")
for lbl, eps, h, cf in (
    ("sim parameters (eps 0.40, h 15)", EPS_SIM, H_AIR, 0.0),
    ("literature oxidised 316L (eps 0.80, h 15)", 0.80, H_AIR, 0.0),
    ("eps 0.80 + forced convection h 50", 0.80, 50.0, 0.0),
    ("eps 0.80 + 2% of surface in die contact", 0.80, H_AIR, 0.02),
    ("eps 0.80 + 5% of surface in die contact", 0.80, H_AIR, 0.05),
):
    r = rate(T_med, eps, h, cf)
    print(f"  {lbl:<42}{r:8.2f}{r/MEASURED:12.2f}x")

print()
print(f"measured (section 3.4 median): {MEASURED:.2f} C/s")
print()
print("across the session's temperature range, sim parameters only:")
for T_c in (615.2, 700.0, 823.6, 900.0, 967.1):
    print(f"  {T_c:7.1f} C -> {rate(T_c, EPS_SIM, H_AIR):5.2f} C/s")

print()
print("what emissivity alone would be needed, with no conduction and h = 15?")
lo, hi = 0.01, 20.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if rate(T_med, mid, H_AIR) < MEASURED:
        lo = mid
    else:
        hi = mid
print(f"  eps = {0.5*(lo+hi):.2f}   (physical maximum is 1.0)")
