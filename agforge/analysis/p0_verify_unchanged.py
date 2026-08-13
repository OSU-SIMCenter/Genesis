"""Phase 0 acceptance test: did the instrumentation change production behaviour?

Closes confound #8 empirically instead of by argument. Every Phase-0 addition is behind a
`qd.static` flag that is False by default, so production *should* compile out unchanged --
but "should" is exactly the kind of claim this project has been burned by.

The control matters: MPM on GPU is not bitwise deterministic here (the 17-hit replay is
known to fail at different hits across seeds), so a nonzero A-vs-baseline difference proves
nothing on its own. What it has to be measured against is the difference between two
BASELINE runs of identical code and config. If new-vs-baseline sits inside
baseline-vs-baseline, the patches are behaviourally invisible.

    A = post-Phase-0 default run
    B, C = two pre-Phase-0 runs, same config (res 10, cfl 0.45, defaults)

Verdict: |A-B| must be comparable to |B-C|, not systematically larger.
"""
import json
import os
import sqlite3
import sys

import numpy as np

OUT = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs")


def load(db, hit):
    con = sqlite3.connect("file:%s?mode=ro" % os.path.join(OUT, db), uri=True)
    r = con.execute("select vertices from hits where step_number=?", (hit,)).fetchone()
    con.close()
    return None if r is None else np.asarray(json.loads(r[0]), dtype=np.float64).reshape(-1, 3)


def stats(P, Q):
    """Per-particle displacement between two clouds with identical particle ordering."""
    d = np.linalg.norm(P - Q, axis=1)
    return d.mean(), d.max(), np.percentile(d, 99)


def main():
    A = "velo_p0_smoke.db"
    B, C = "velo_dlv_c10_r1.db", "velo_dlv_c10_r2.db"
    print("A (post-Phase-0) = %s" % A)
    print("B, C (pre-Phase-0 baselines, identical config) = %s, %s" % (B, C))
    print()
    hdr = "%-6s %-26s %12s %12s %12s"
    print(hdr % ("hit", "comparison", "mean |dx| mm", "p99 mm", "max mm"))
    print("-" * 72)
    verdict = []
    for hit in (1, 2):
        a, b, c = load(A, hit), load(B, hit), load(C, hit)
        if a is None or b is None or c is None:
            print("hit %d: missing in one db" % hit)
            continue
        if not (a.shape == b.shape == c.shape):
            print("hit %d: SHAPE MISMATCH %s %s %s" % (hit, a.shape, b.shape, c.shape))
            continue
        ab = stats(a, b)
        bc = stats(b, c)
        ac = stats(a, c)
        print(hdr % (hit, "new vs baseline1 (A-B)", "%.6f" % ab[0], "%.6f" % ab[2], "%.6f" % ab[1]))
        print(hdr % ("", "new vs baseline2 (A-C)", "%.6f" % ac[0], "%.6f" % ac[2], "%.6f" % ac[1]))
        print(hdr % ("", "BASELINE NOISE  (B-C)", "%.6f" % bc[0], "%.6f" % bc[2], "%.6f" % bc[1]))
        print()
        verdict.append((hit, ab[0], ac[0], bc[0]))

    print("=" * 72)
    for hit, ab, ac, bc in verdict:
        if bc == 0.0:
            ok = (ab == 0.0 and ac == 0.0)
            note = "baseline is bitwise deterministic; new run must match exactly"
        else:
            ok = ab <= 3.0 * bc and ac <= 3.0 * bc
            note = "new-vs-baseline within 3x the baseline noise"
        print("hit %d: %s  (%s)" % (hit, "PASS" if ok else "FAIL -- INVESTIGATE", note))
        print("        A-B %.3e   A-C %.3e   B-C %.3e" % (ab, ac, bc))


if __name__ == "__main__":
    sys.exit(main())
