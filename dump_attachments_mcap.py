#!/usr/bin/env python3
"""Write every MCAP attachment out to a directory, byte-for-byte."""
import os
import struct
import sys


class R:
    def __init__(s, b): s.b = b; s.i = 0
    def u16(s): v = struct.unpack_from("<H", s.b, s.i)[0]; s.i += 2; return v
    def u32(s): v = struct.unpack_from("<I", s.b, s.i)[0]; s.i += 4; return v
    def u64(s): v = struct.unpack_from("<Q", s.b, s.i)[0]; s.i += 8; return v
    def s_(s):
        n = s.u32(); v = s.b[s.i:s.i + n].decode("utf-8", "replace"); s.i += n; return v


def records(buf):
    i, n = 0, len(buf)
    while i + 9 <= n:
        op = buf[i]
        ln = struct.unpack_from("<Q", buf, i + 1)[0]
        st, en = i + 9, i + 9 + ln
        if en > n:
            break
        yield op, buf[st:en]
        i = en


def main(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(size - 8 - 29)
        foot = f.read(8 + 29)[:29]
        ss, _so, _c = struct.unpack_from("<QQI", foot, 9)
        f.seek(ss)
        summary = f.read((size - 8 - 29) - ss)
        idx = []
        for op, c in records(summary):
            if op == 0x0A:
                r = R(c)
                off = r.u64(); ln = r.u64(); r.u64(); r.u64(); r.u64()
                nm = r.s_(); mt = r.s_()
                idx.append((off, ln, nm, mt))
        for off, ln, nm, mt in idx:
            f.seek(off)
            rec = f.read(ln)
            rr = R(rec[9:])
            rr.u64(); rr.u64()
            name = rr.s_(); rr.s_()
            dlen = rr.u64()
            body = rr.b[rr.i:rr.i + dlen]
            dest = os.path.join(outdir, os.path.basename(name))
            with open(dest, "wb") as o:
                o.write(body)
            print(f"wrote {dest}  ({len(body):,} B, declared {dlen:,})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
