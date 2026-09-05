"""Full pairwise distance matrix over the banked and post-port runs.

WHY THIS REPLACES rt4's VERDICT: rt4 estimated the run-to-run noise floor from a SINGLE pair
(N1 vs N2). That is one sample of a random variable, and it then reported a 5.20x ratio against
it as if the denominator were solid. With three banked runs available there are three independent
same-config pairs, so the floor can be given a range instead of a point -- and a 5x excursion
against a point estimate often stops being remarkable once the spread is known.

The matrix also answers a question a ratio cannot: do the post-port runs (S/W/P) cluster with each
other and sit at the banked cluster's own spread, or do they sit apart from the banked runs as a
group? The latter would indicate a real behavioural shift somewhere between when the banked runs
were made and now -- which is NOT necessarily this port, since another agent changed
replay_episode.py (171 lines) in the same window.
"""
import itertools
import json
import os
import sqlite3

import numpy as np

OUT = os.path.expanduser("~/GitHub/Genesis/forge_common/main/outputs")

BANKED = {
    "N1": "velo_mx_grid_position_correction_r1.db",
    "N2": "velo_mx_grid_position_correction_r2.db",
    "N3": "velo_mx_grid_position_correction_r3.db",
}
NEW = {
    "S": "velo_rt_static.db",
    "W": "velo_rt_switch.db",
    "P": "velo_rt_probe.db",
}
ALL = {**BANKED, **NEW}


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


def main():
    for hit in (1, 2):
        print("=" * 76)
        print("HIT %d -- mean per-particle |dx|" % hit)
        print("=" * 76)
        P = {}
        for k, db in ALL.items():
            v = load(db, hit)
            if v is not None:
                P[k] = v
        keys = [k for k in ALL if k in P]
        if len(keys) < 2:
            print("  not enough runs loaded")
            continue

        shapes = {k: P[k].shape for k in keys}
        if len(set(shapes.values())) != 1:
            print("  SHAPE MISMATCH: %s" % shapes)
            continue

        d = {}
        for a, b in itertools.combinations(keys, 2):
            d[(a, b)] = float(np.linalg.norm(P[a] - P[b], axis=1).mean())

        print("      " + "".join("%10s" % k for k in keys))
        for a in keys:
            row = "%-6s" % a
            for b in keys:
                if a == b:
                    row += "%10s" % "-"
                else:
                    row += "%10.2e" % d[tuple(sorted((a, b), key=keys.index))]
            print(row)
        print()

        banked_pairs = [d[p] for p in d if p[0] in BANKED and p[1] in BANKED]
        new_pairs = [d[p] for p in d if p[0] in NEW and p[1] in NEW]
        cross = [d[p] for p in d if (p[0] in BANKED) != (p[1] in BANKED)]

        def summ(name, vals):
            if not vals:
                print("  %-28s (none)" % name)
                return None
            print("  %-28s n=%d  min %.2e  max %.2e  mean %.2e"
                  % (name, len(vals), min(vals), max(vals), sum(vals) / len(vals)))
            return max(vals)

        bmax = summ("banked-vs-banked (the floor)", banked_pairs)
        summ("new-vs-new", new_pairs)
        cmax = summ("banked-vs-new (cross)", cross)
        if bmax and cross:
            print()
            print("  cross/floor using the floor's OWN MAX (not a single pair): %.2fx"
                  % (max(cross) / bmax))
            if max(cross) <= bmax:
                print("  => post-port runs sit INSIDE the banked runs' own spread.")
            else:
                print("  => post-port runs sit OUTSIDE the banked spread. Something changed")
                print("     between the banked runs and now; this port is only one candidate")
                print("     (replay_episode.py also changed by ~171 lines in that window).")
        print()


if __name__ == "__main__":
    main()
