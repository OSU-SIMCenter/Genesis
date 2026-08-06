#!/usr/bin/env python3
"""Extract the JSON telemetry channels from the T4 bulk MCAP into a compact NPZ.

Thermal frames are ~99% of the file by volume and are skipped without parsing;
only the small JSON topics are decoded. One pass, cached to disk, so the
downstream analysis never has to touch the 8.58 GB file again.
"""
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

MAGIC = b"\x89MCAP0\r\n"


class R:
    def __init__(s, b): s.b = b; s.i = 0
    def u16(s): v = struct.unpack_from("<H", s.b, s.i)[0]; s.i += 2; return v
    def u32(s): v = struct.unpack_from("<I", s.b, s.i)[0]; s.i += 4; return v
    def u64(s): v = struct.unpack_from("<Q", s.b, s.i)[0]; s.i += 8; return v
    def s_(s):
        n = s.u32(); v = s.b[s.i:s.i + n].decode("utf-8", "replace"); s.i += n; return v
    def by(s):
        n = s.u32(); v = s.b[s.i:s.i + n]; s.i += n; return v


def records(buf):
    i, n = 0, len(buf)
    while i + 9 <= n:
        op = buf[i]
        length = struct.unpack_from("<Q", buf, i + 1)[0]
        st = i + 9
        en = st + length
        if en > n:
            break
        yield op, buf[st:en]
        i = en


def zd(data):
    p = subprocess.run(["zstd", "-d", "-c"], input=data,
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return p.stdout


def main(path, out):
    size = os.path.getsize(path)
    t_start = time.time()
    with open(path, "rb") as f:
        f.seek(size - 8 - 29)
        foot = f.read(8 + 29)[:29]
        summary_start, _soff, _crc = struct.unpack_from("<QQI", foot, 9)
        f.seek(summary_start)
        summary = f.read((size - 8 - 29) - summary_start)

        channels = {}
        chunks = []
        for op, c in records(summary):
            r = R(c)
            if op == 0x04:
                cid = r.u16(); r.u16(); topic = r.s_(); menc = r.s_()
                channels[cid] = (topic, menc)
            elif op == 0x08:
                r.u64(); r.u64(); cso = r.u64(); cl = r.u64()
                chunks.append((cso, cl))
        chunks.sort()

        want = {cid: t for cid, (t, enc) in channels.items() if enc == "json"}
        print(f"JSON channels: {sorted(want.values())}")
        print(f"chunks: {len(chunks):,}")

        press_t, press_f, press_s, press_p = [], [], [], []
        press_flags = []          # (is_idle, ready, cycle_end, pass, end)
        utaken = []               # (log_time_ns, obj)
        torm = []                 # (log_time_ns, actual_position[9])
        ard = []                  # (log_time_ns, ram_linear_position, air_pressure, heater_on)

        for k, (cso, cl) in enumerate(chunks):
            f.seek(cso)
            rec = f.read(cl)
            r = R(rec[9:])
            r.u64(); r.u64(); r.u64(); r.u32()
            comp = r.s_(); rl = r.u64()
            raw = zd(r.b[r.i:r.i + rl]) if comp == "zstd" else r.b[r.i:r.i + rl]

            for op, c in records(raw):
                if op != 0x05:
                    continue
                cid = struct.unpack_from("<H", c, 0)[0]
                topic = want.get(cid)
                if topic is None:
                    continue                      # thermal frames: never parsed
                log_time = struct.unpack_from("<Q", c, 6)[0]
                try:
                    o = json.loads(c[22:].decode("utf-8", "replace"))
                except Exception:
                    continue
                if topic == "hmr/press/state":
                    press_t.append(log_time)
                    press_f.append(o.get("live_force_kn", np.nan))
                    press_s.append(o.get("live_stroke_mm", np.nan))
                    press_p.append(o.get("live_position_mm", np.nan))
                    press_flags.append((
                        bool(o.get("is_idle", False)), bool(o.get("ready", False)),
                        bool(o.get("cycle_end", False)), bool(o.get("pass", False)),
                        bool(o.get("end", False)),
                    ))
                elif topic == "forge/deform/u_taken":
                    utaken.append((log_time, o))
                elif topic == "hmr/torm/state":
                    ap = o.get("actual_position") or []
                    if len(ap) >= 9:
                        torm.append((log_time, *ap[:9]))
                elif topic == "hmr/ard/state":
                    an = o.get("analog") or {}
                    dg = o.get("digital") or {}
                    ard.append((log_time,
                                an.get("ram_linear_position", np.nan),
                                an.get("air_pressure", np.nan),
                                bool(dg.get("heater_on", False))))

            if (k + 1) % 100 == 0 or k + 1 == len(chunks):
                el = time.time() - t_start
                print(f"  chunk {k+1:,}/{len(chunks):,}  press={len(press_t):,}  "
                      f"u_taken={len(utaken)}  {el:.0f}s", flush=True)

    np.savez_compressed(
        out,
        press_t=np.array(press_t, dtype=np.uint64),
        press_force_kn=np.array(press_f, dtype=np.float64),
        press_stroke_mm=np.array(press_s, dtype=np.float64),
        press_position_mm=np.array(press_p, dtype=np.float64),
        press_flags=np.array(press_flags, dtype=bool),
        torm=np.array(torm, dtype=np.float64) if torm else np.zeros((0, 10)),
        ard=np.array(ard, dtype=np.float64) if ard else np.zeros((0, 4)),
    )
    with open(os.path.splitext(out)[0] + "_utaken.json", "w") as fh:
        json.dump([{"log_time_ns": t, **o} for t, o in utaken], fh, indent=1)

    print(f"\npress samples : {len(press_t):,}")
    print(f"u_taken       : {len(utaken)}")
    print(f"torm          : {len(torm):,}")
    print(f"ard           : {len(ard):,}")
    print(f"wrote {out} ({os.path.getsize(out):,} B) in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
