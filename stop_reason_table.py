#!/usr/bin/env python3
"""Print the per-hit press stop table for a run log.

The cheap test §4.7.3 prescribes. For any run log, says which hits reached their
commanded strain and which were truncated by the `Max Force` control stop, and
pairs each against the contact width it pressed at.

    python3 stop_reason_table.py ~/profile_g0_17.log

Reads the two lines the adapter and controller already emit:

    [genesis adapter] PRESSING target_strain=0.2437 (W_contact=63.600mm, ...)
    Strike -> HOLDING (Target Strain, strain=0.2437, steps=68, time=1.91s)

A run whose stop reasons are all `Target Strain` was never force-limited and its
geometry is comparable across configurations; one with `Max Force` hits is
truncated on those hits and is not. Note that the threshold itself is no longer a
single constant across worktrees -- `nsf-demo` reads AGF_MAX_FORCE from the
environment -- so record which worktree and environment produced the log.
"""
import re
import sys

CMD = re.compile(r"target_strain=([0-9.]+)\s*\(W_contact=([0-9.]+)mm")
GOT = re.compile(r"HOLDING \(([A-Za-z ]+), strain=([0-9.]+)")


def main(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    cmds, gots = CMD.findall(text), GOT.findall(text)
    if not gots:
        print("no 'Strike -> HOLDING' lines in %s" % path)
        return 1
    if len(cmds) != len(gots):
        print("! %d commanded vs %d completed - run ended mid-hit; pairing the "
              "first %d" % (len(cmds), len(gots), min(len(cmds), len(gots))))

    print("%-4s %-10s %-10s %-10s %-10s %s"
          % ("hit", "W_contact", "commanded", "achieved", "shortfall", "stop"))
    saturated = 0
    for i, ((cmd, w), (reason, got)) in enumerate(zip(cmds, gots), start=1):
        short = float(cmd) - float(got)
        if reason.strip() == "Max Force":
            saturated += 1
        print("%-4d %-10s %-10s %-10s %-10s %s"
              % (i, w, cmd, got, "%.4f" % short if short > 0.0005 else "-", reason))

    n = len(gots)
    print("\n%d of %d hits truncated by the force stop (%.0f%%); %d reached command."
          % (saturated, n, 100.0 * saturated / n, n - saturated))
    if saturated:
        narrow = [float(w) for (_, w), (r, _) in zip(cmds, gots) if r.strip() == "Max Force"]
        wide = [float(w) for (_, w), (r, _) in zip(cmds, gots) if r.strip() != "Max Force"]
        if narrow and wide:
            print("mean W_contact: %.1f mm when truncated vs %.1f mm when not."
                  % (sum(narrow) / len(narrow), sum(wide) / len(wide)))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
