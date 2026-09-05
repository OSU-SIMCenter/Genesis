"""Did the runtime-switchable port change what the simulation actually does?

Same method as p0_verify_unchanged.py, and for the same reason: this sim is NOT bitwise
deterministic on GPU (measured 0.4-1.7 um mean particle displacement between identical runs), so a
nonzero difference between two runs proves nothing by itself. It only means something measured
against the run-to-run noise floor of identical code and config.

    S = post-port, static build      (AGF_CONTACT_RUNTIME_SWITCH=0)
    W = post-port, switchable build  (AGF_CONTACT_RUNTIME_SWITCH=1)
    P = post-port, switchable + penetration probe

    N1, N2 = two banked grid+position-correction runs -- identical config, so |N1-N2| IS the noise floor

Three claims under test, each of which was asserted in a code comment before being measured:
    1. the port left the static path alone      -> |S-N| ~ |N1-N2|
    2. switchable behaves like static           -> |S-W| ~ |N1-N2|
    3. the penetration probe only reads state   -> |W-P| ~ |N1-N2|
"""
import json
import os
import sqlite3
import sys

import numpy as np

OUT = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs")


def load(db, hit):
    path = os.path.join(OUT, db)
    if not os.path.exists(path):
        return None
    con = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    try:
        r = con.execute("select vertices from hits where step_number=?", (hit,)).fetchone()
    except Exception:
        return None
    finally:
        con.close()
    return None if r is None else np.asarray(json.loads(r[0]), dtype=np.float64).reshape(-1, 3)


def stats(P, Q):
    d = np.linalg.norm(P - Q, axis=1)
    return d.mean(), np.percentile(d, 99), d.max()


PAIRS = [
    ("S vs N1   (port vs banked)", "velo_rt_static.db", "velo_mx_grid_position_correction_r1.db"),
    ("S vs W    (static vs switchable)", "velo_rt_static.db", "velo_rt_switch.db"),
    ("W vs P    (probe inert?)", "velo_rt_switch.db", "velo_rt_probe.db"),
    ("NOISE FLOOR N1 vs N2", "velo_mx_grid_position_correction_r1.db", "velo_mx_grid_position_correction_r2.db"),
]


def main():
    hdr = "%-36s %14s %14s %14s"
    any_rows = False
    for hit in (1, 2):
        print("=" * 84)
        print("HIT %d" % hit)
        print("=" * 84)
        print(hdr % ("comparison", "mean |dx|", "p99", "max"))
        print("-" * 84)
        floor = None
        rows = []
        for label, a_db, b_db in PAIRS:
            A, B = load(a_db, hit), load(b_db, hit)
            if A is None or B is None:
                print("%-36s  MISSING (%s / %s)" % (label, a_db if A is None else "", b_db if B is None else ""))
                continue
            if A.shape != B.shape:
                print("%-36s  SHAPE MISMATCH %s vs %s" % (label, A.shape, B.shape))
                continue
            m, p99, mx = stats(A, B)
            print(hdr % (label, "%.3e" % m, "%.3e" % p99, "%.3e" % mx))
            rows.append((label, m))
            if label.startswith("NOISE"):
                floor = m
            any_rows = True
        print()
        if floor is not None and floor > 0:
            print("  verdict (mean |dx| as a multiple of the noise floor):")
            for label, m in rows:
                if label.startswith("NOISE"):
                    continue
                ratio = m / floor
                tag = "OK -- within noise" if ratio <= 3.0 else "!! LARGER THAN NOISE -- investigate"
                print("    %-36s %6.2fx   %s" % (label, ratio, tag))
        elif floor == 0:
            print("  noise floor is exactly 0 -- runs are bitwise identical here; any nonzero")
            print("  difference above is real and must be explained, not averaged away.")
        else:
            print("  no noise floor available (banked runs missing) -- ratios not computable.")
        print()

    if not any_rows:
        print("No comparable rows found. Check that the rt runs completed and wrote hits.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
