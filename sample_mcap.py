#!/usr/bin/env python3
"""Sample real message payloads + attachments from a (zstd) MCAP file.

Uses the `zstd` CLI for chunk decompression (no python zstd module needed).
"""
import struct, sys, os, json, subprocess

MAGIC = b"\x89MCAP0\r\n"


class R:
    def __init__(s, b): s.b = b; s.i = 0
    def u8(s):  v = s.b[s.i]; s.i += 1; return v
    def u16(s): v = struct.unpack_from("<H", s.b, s.i)[0]; s.i += 2; return v
    def u32(s): v = struct.unpack_from("<I", s.b, s.i)[0]; s.i += 4; return v
    def u64(s): v = struct.unpack_from("<Q", s.b, s.i)[0]; s.i += 8; return v
    def s_(s):
        n = s.u32(); v = s.b[s.i:s.i+n].decode("utf-8", "replace"); s.i += n; return v
    def by(s):
        n = s.u32(); v = s.b[s.i:s.i+n]; s.i += n; return v


def records(buf):
    i, n = 0, len(buf)
    while i + 9 <= n:
        op = buf[i]
        length = struct.unpack_from("<Q", buf, i+1)[0]
        start = i + 9; end = start + length
        if end > n: break
        yield op, buf[start:end]
        i = end


def zstd_decompress(data):
    p = subprocess.run(["zstd", "-d", "-c"], input=data,
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return p.stdout


def parse_protobuf_rawimage(data):
    """Minimal foxglove.RawImage field extraction."""
    out = {}
    i, n = 0, len(data)
    def varint(i):
        shift = 0; val = 0
        while True:
            b = data[i]; i += 1
            val |= (b & 0x7F) << shift
            if not (b & 0x80): break
            shift += 7
        return val, i
    while i < n:
        key, i = varint(i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(i)
            if fn == 3: out["width"] = v
            elif fn == 4: out["height"] = v
            elif fn == 6: out["step"] = v
        elif wt == 2:
            ln, i = varint(i)
            chunk = data[i:i+ln]; i += ln
            if fn == 2: out["frame_id"] = chunk.decode("utf-8", "replace")
            elif fn == 5: out["encoding"] = chunk.decode("utf-8", "replace")
            elif fn == 7: out["data_len"] = len(chunk)
            elif fn == 1:  # timestamp submessage
                pass
        elif wt == 1: i += 8
        elif wt == 5: i += 4
        else: break
    return out


def main(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(size - 8 - 29)
        tail = f.read(8 + 29)
        foot = tail[:29]
        summary_start, soff, crc = struct.unpack_from("<QQI", foot, 9)
        f.seek(summary_start)
        summary = f.read((size - 8 - 29) - summary_start)

        schemas, channels = {}, {}
        chunk_offsets = []   # (offset, length)
        att_index = []       # (offset, length, name, media)
        for op, c in records(summary):
            r = R(c)
            if op == 0x03:
                sid = r.u16(); nm = r.s_(); enc = r.s_(); dat = r.by()
                schemas[sid] = (nm, enc)
            elif op == 0x04:
                cid = r.u16(); sid = r.u16(); topic = r.s_(); menc = r.s_()
                channels[cid] = (topic, menc, sid)
            elif op == 0x08:  # ChunkIndex
                _ms = r.u64(); _me = r.u64(); cso = r.u64(); cl = r.u64()
                chunk_offsets.append((cso, cl))
            elif op == 0x0A:  # AttachmentIndex
                off = r.u64(); ln = r.u64(); _lt = r.u64(); _ct = r.u64()
                _ds = r.u64(); nm = r.s_(); mt = r.s_()
                att_index.append((off, ln, nm, mt))

        # -------- attachments --------
        print("=" * 78)
        print("ATTACHMENTS (full content)")
        print("=" * 78)
        for off, ln, nm, mt in att_index:
            f.seek(off)
            rec = f.read(ln)
            # rec: op(1) len(8) [log_time8 create8 nameStr mediaStr datalen8 data crc4]
            rr = R(rec[9:])
            _lt = rr.u64(); _ct = rr.u64(); name = rr.s_(); media = rr.s_()
            dlen = rr.u64(); body = rr.b[rr.i:rr.i+dlen]
            print(f"\n----- {name}  ({media}, {dlen} bytes) -----")
            txt = body.decode("utf-8", "replace")
            print(txt)

        # -------- first chunk -> sample one msg per channel --------
        chunk_offsets.sort()
        # decode several chunks until we have at least one msg per channel
        seen = {}
        for (cso, cl) in chunk_offsets[:60]:
            f.seek(cso)
            rec = f.read(cl)
            r = R(rec[9:])  # skip op+len of chunk record
            _ms = r.u64(); _me = r.u64(); usize = r.u64(); ucrc = r.u32()
            comp = r.s_(); reclen = r.u64()
            comp_data = r.b[r.i:r.i+reclen]
            raw = zstd_decompress(comp_data) if comp == "zstd" else comp_data
            for op, c in records(raw):
                if op == 0x05:  # Message
                    cid = struct.unpack_from("<H", c, 0)[0]
                    if cid in seen: continue
                    data = c[22:]
                    seen[cid] = data
            if len(seen) >= len(channels):
                break

        print("\n" + "=" * 78)
        print("SAMPLE MESSAGE PER TOPIC")
        print("=" * 78)
        for cid, (topic, menc, sid) in channels.items():
            sname = schemas.get(sid, ("<none>", ""))[0]
            print(f"\n----- {topic}  (encoding={menc}, schema={sname}) -----")
            data = seen.get(cid)
            if data is None:
                print("   (no sample found in scanned chunks)")
                continue
            if menc == "json":
                try:
                    obj = json.loads(data.decode("utf-8", "replace"))
                    print(json.dumps(obj, indent=2)[:4000])
                except Exception as e:
                    print("   JSON parse error:", e)
                    print("   raw:", data[:500])
            elif menc == "protobuf":
                info = parse_protobuf_rawimage(data)
                print("   RawImage fields:", json.dumps(info, indent=2))
                print(f"   (serialized message size: {len(data):,} bytes)")
            else:
                print("   raw bytes:", data[:200])


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "20260615_180456_T4_bulk.mcap"
    main(p)
