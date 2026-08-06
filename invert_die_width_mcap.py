#!/usr/bin/env python3
"""Invert the measured blow for the die contact length.

Force comparison is meaningless without contact area, and the die width for
tool 2 is not documented. But the loading curve constrains it: for a cylinder
of radius R squeezed between flat dies to gap h, the contact half-width is
    a = sqrt(R^2 - (h/2)^2)
and the force is
    F = p * (2a * L_die),   p = constraint * sigma_flow(eps)
Everything but L_die is known, so solve for it at every sample. If the model
is right, L_die comes out CONSTANT across the whole loading curve -- that
constancy is the test, not the value.

sigma_flow uses the committed 316L Johnson-Cook card, which (with T* clamped
at T_ref = 1273.15 K) is the 1000 C / 1 s^-1 flow stress:
    sigma = A + B * eps^n  =  100.3 + 195.0 * eps^0.417   [MPa]
"""
import sys

import numpy as np

R_MM = 19.05          # 38.1 mm billet
JC_A, JC_B, JC_N = 100.3, 195.0, 0.417


def main(blow_npz):
    d = np.load(blow_npz)
    t, f, h = d["t"], d["force_kn"], d["position_mm"]

    # Loading branch: contact (force rises off baseline) to peak.
    ipk = int(np.argmax(f))
    contact = np.flatnonzero(f[:ipk] < 5.0)
    i0 = contact[-1] + 1 if len(contact) else 0
    print(f"contact at t={t[i0]:.3f} s, gap={h[i0]:.3f} mm, F={f[i0]:.2f} kN")
    print(f"peak    at t={t[ipk]:.3f} s, gap={h[ipk]:.3f} mm, F={f[ipk]:.2f} kN")
    h0 = h[i0]
    print(f"=> billet diameter from contact gap: {h0:.2f} mm (nominal 38.10)\n")

    print(f"{'gap':>7} {'a_mm':>7} {'eps':>7} {'sig_MPa':>8} {'F_kN':>7} "
          f"{'L(c=1.0)':>9} {'L(c=1.5)':>9} {'L(c=2.0)':>9}")
    print("-" * 74)
    rows = []
    for i in range(i0, ipk + 1):
        gap = h[i]
        half = gap / 2.0
        if half >= R_MM:
            continue
        a = np.sqrt(R_MM**2 - half**2)          # contact half-width, mm
        eps = np.log(h0 / gap)                   # true strain (height reduction)
        sig = JC_A + JC_B * max(eps, 1e-6) ** JC_N   # MPa
        area = 2 * a                             # mm^2 per mm of die length
        Ls = []
        for c in (1.0, 1.5, 2.0):
            p = c * sig                          # MPa = N/mm^2
            L = f[i] * 1000.0 / (p * area)       # mm
            Ls.append(L)
        rows.append((gap, a, eps, sig, f[i], *Ls))
    step = max(1, len(rows) // 22)
    for r in rows[::step]:
        print(f"{r[0]:7.2f} {r[1]:7.2f} {r[2]:7.4f} {r[3]:8.1f} {r[4]:7.2f} "
              f"{r[5]:9.2f} {r[6]:9.2f} {r[7]:9.2f}")

    arr = np.array(rows)
    # judge constancy over the well-developed part of the curve (eps > 0.05)
    dev = arr[arr[:, 2] > 0.05]
    print()
    for j, c in ((5, 1.0), (6, 1.5), (7, 2.0)):
        L = dev[:, j]
        print(f"constraint {c:.1f}: L_die = {L.mean():6.2f} +/- {L.std():.2f} mm "
              f"(spread {L.min():.2f}-{L.max():.2f}, "
              f"CV {L.std()/L.mean()*100:.1f}%)")
    print()
    print("A LOW CV means force scales with contact width exactly as flat-die")
    print("indentation predicts -- i.e. the geometry model is right and L_die")
    print("is identified. A HIGH CV means something else is going on.")


if __name__ == "__main__":
    main(sys.argv[1])
