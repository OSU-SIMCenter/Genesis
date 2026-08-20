"""Does the die-balance controller confound the D1a press-speed sweep?

The stall bound |dF|_stall = sqrt(pressing_speed * 20000 / gain) carries a sqrt of press
speed, so a speed sweep lowers it. That predicts MORE stalling at slow speed. This measures
whether that actually happened, from the sweep's own logs.

Crucially the logs record v=[vL,vR], so stalling is OBSERVED (a die commanded to exactly
zero) rather than inferred from the bound. Both are reported.

Usage: python3 stall_audit.py <log> [<log> ...]
"""
import math
import re
import sys
from collections import OrderedDict

PRESSING = re.compile(
    r"PRESSING\[(\d+)\]: F=\[([-0-9.]+),([-0-9.]+)\] dF=([-0-9.]+), v=\[([-0-9.]+),([-0-9.]+)\]")
NEW_HIT = re.compile(r"Strike -> PRESSING \(width=")
SPEED = re.compile(r"pressing_speed=([0-9.]+)\s*m/s")
ARM = re.compile(r"^ARM\s+\d+/\d+\s+(\S+)")

SAFETY = 20000.0
GAIN = 1.5e-4


def bound(v, gain=GAIN):
    return math.sqrt(v * SAFETY / gain)


def scan(path):
    seg = OrderedDict()
    speed = arm = None
    hit_had_stall = False
    for line in open(path, encoding="utf-8", errors="replace"):
        m = SPEED.search(line)
        if m and "###" in line:
            speed = float(m.group(1))
            continue
        m = ARM.search(line.strip())
        if m:
            arm = m.group(1)
            continue
        if NEW_HIT.search(line):
            if speed is not None and arm is not None:
                seg.setdefault((speed, arm), _blank())["hits"] += 1
            hit_had_stall = False
            continue
        m = PRESSING.search(line)
        if not m or speed is None or arm is None:
            continue
        d = seg.setdefault((speed, arm), _blank())
        dF = abs(float(m.group(4)))
        vL, vR = float(m.group(5)), float(m.group(6))
        d["frames"] += 1
        d["sum_dF"] += dF
        d["max_dF"] = max(d["max_dF"], dF)
        if vL == 0.0 or vR == 0.0:
            d["stall_frames"] += 1
            if not hit_had_stall:
                d["stall_hits"] += 1
                hit_had_stall = True
        if dF > bound(speed):
            d["over_frames"] += 1
        if dF > SAFETY:
            d["modul_frames"] += 1
    return seg


def _blank():
    return {"frames": 0, "stall_frames": 0, "over_frames": 0, "modul_frames": 0,
            "hits": 0, "stall_hits": 0, "sum_dF": 0.0, "max_dF": 0.0}


def main(paths):
    for path in paths:
        seg = scan(path)
        print("=" * 104)
        print(path)
        print("=" * 104)
        if not seg:
            print("  no PRESSING telemetry parsed")
            continue
        print("%-9s %-20s %7s %9s %9s %10s %10s %11s %10s"
              % ("v (m/s)", "arm", "hits", "frames", "bound N", "modul>20k",
                 "pred>bound", "OBS stall", "max|dF| N"))
        print("-" * 104)
        for (v, arm) in sorted(seg, key=lambda k: (-k[0], k[1])):
            d = seg[(v, arm)]
            f = d["frames"] or 1
            print("%-9.3f %-20s %7d %9d %9.0f %9.1f%% %9.1f%% %6.1f%% %2d/%-2dh %10.0f"
                  % (v, arm, d["hits"], d["frames"], bound(v),
                     100.0 * d["modul_frames"] / f,
                     100.0 * d["over_frames"] / f,
                     100.0 * d["stall_frames"] / f, d["stall_hits"], d["hits"],
                     d["max_dF"]))
        print()
        arms = sorted({a for (_, a) in seg})
        for arm in arms:
            pts = sorted([(v, seg[(v, a)]) for (v, a) in seg if a == arm], reverse=True)
            if len(pts) < 2:
                continue
            trend = " ".join("%.3g:%.1f%%" % (v, 100.0 * d["stall_frames"] / (d["frames"] or 1))
                             for v, d in pts)
            print("  OBSERVED stall%% vs speed  %-20s %s" % (arm, trend))
        print()


main(sys.argv[1:] or ["/home/timothy/speed_sweep.log"])
