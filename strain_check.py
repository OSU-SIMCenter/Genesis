"""Were the D1a speed points compared at EQUAL DEFORMATION?

W-A found that a force-truncated sweep inverts the sign, because the fast run stops at
lower strain than the slow one and the two are then compared at different deformation.
That artifact needs unequal final strain across speed points. This checks directly.

Usage: python3 strain_check.py <log> [<log> ...]
"""
import re
import sys
from collections import OrderedDict

HOLDING = re.compile(r"Strike -> HOLDING \(([A-Za-z ]+), strain=([0-9.]+)")
SPEED = re.compile(r"pressing_speed=([0-9.]+)\s*m/s")
ARM = re.compile(r"^ARM\s+\d+/\d+\s+(\S+)")


def main(paths):
    for path in paths:
        rows = OrderedDict()
        speed = arm = None
        for line in open(path, encoding="utf-8", errors="replace"):
            m = SPEED.search(line)
            if m and "###" in line:
                speed = float(m.group(1))
                continue
            m = ARM.search(line.strip())
            if m:
                arm = m.group(1)
                continue
            m = HOLDING.search(line)
            if m and speed is not None and arm is not None:
                rows.setdefault((speed, arm), []).append((m.group(1).strip(),
                                                          float(m.group(2))))
        print("=" * 92)
        print(path)
        print("=" * 92)
        if not rows:
            print("  no HOLDING lines")
            continue
        print("%-9s %-20s %6s %9s %9s %9s  %s"
              % ("v (m/s)", "arm", "hits", "mean eps", "min", "max", "stop reasons"))
        print("-" * 92)
        by_speed = {}
        for (v, a) in sorted(rows, key=lambda k: (-k[0], k[1])):
            vals = [e for _, e in rows[(v, a)]]
            stops = sorted({s for s, _ in rows[(v, a)]})
            by_speed.setdefault(v, []).extend(vals)
            print("%-9.3f %-20s %6d %9.4f %9.4f %9.4f  %s"
                  % (v, a, len(vals), sum(vals) / len(vals), min(vals), max(vals),
                     ", ".join(stops)))
        print()
        print("  per-speed mean strain (the quantity the artifact would move):")
        base = None
        for v in sorted(by_speed, reverse=True):
            vals = by_speed[v]
            mean = sum(vals) / len(vals)
            if base is None:
                base = mean
            print("    v=%-7.3f  n=%-4d mean eps=%.4f   delta vs fastest = %+.4f"
                  % (v, len(vals), mean, mean - base))
        print()


main(sys.argv[1:])
