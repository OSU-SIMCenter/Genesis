#!/usr/bin/env python3
"""Self-contained MCAP summary inspector (no third-party deps).

Reads only the file header and the trailing summary/index section, so it works
on huge files (GBs) without streaming the whole thing.
"""
import struct
import sys
import os

MAGIC = b"\x89MCAP0\r\n"

OP = {
    0x01: "Header", 0x02: "Footer", 0x03: "Schema", 0x04: "Channel",
    0x05: "Message", 0x06: "Chunk", 0x07: "MessageIndex", 0x08: "ChunkIndex",
    0x09: "Attachment", 0x0A: "AttachmentIndex", 0x0B: "Statistics",
    0x0C: "Metadata", 0x0D: "MetadataIndex", 0x0E: "SummaryOffset", 0x0F: "DataEnd",
}


class Reader:
    def __init__(self, b):
        self.b = b
        self.i = 0

    def u8(self):
        v = self.b[self.i]; self.i += 1; return v

    def u16(self):
        v = struct.unpack_from("<H", self.b, self.i)[0]; self.i += 2; return v

    def u32(self):
        v = struct.unpack_from("<I", self.b, self.i)[0]; self.i += 4; return v

    def u64(self):
        v = struct.unpack_from("<Q", self.b, self.i)[0]; self.i += 8; return v

    def s(self):
        n = self.u32()
        v = self.b[self.i:self.i + n].decode("utf-8", "replace"); self.i += n; return v

    def by(self):
        n = self.u32()
        v = self.b[self.i:self.i + n]; self.i += n; return v


def read_records(buf):
    """Yield (opcode, content_bytes) for top-level records in buf."""
    i = 0
    n = len(buf)
    while i + 9 <= n:
        op = buf[i]
        length = struct.unpack_from("<Q", buf, i + 1)[0]
        start = i + 9
        end = start + length
        if end > n:
            break
        yield op, buf[start:end]
        i = end


def main(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        # ---- start magic + header ----
        head = f.read(4096)
        assert head[:8] == MAGIC, "Not an MCAP file (bad start magic)"

        profile = library = "?"
        for op, content in read_records(head[8:]):
            if op == 0x01:  # Header
                r = Reader(content)
                profile = r.s()
                library = r.s()
            break  # header is first record

        # ---- footer (last 8 magic + 29-byte footer record) ----
        f.seek(size - 8 - 29)
        tail = f.read(8 + 29)
        assert tail[-8:] == MAGIC, "Not an MCAP file (bad end magic)"
        foot = tail[:29]
        # footer record: op(1) len(8) summary_start(8) summary_offset_start(8) crc(4)
        assert foot[0] == 0x02, "Footer opcode mismatch"
        summary_start, summary_offset_start, summary_crc = struct.unpack_from(
            "<QQI", foot, 9)

        if summary_start == 0:
            print("No summary section present in this file.")
            return

        # ---- read whole summary section ----
        f.seek(summary_start)
        summary_len = (size - 8 - 29) - summary_start  # up to footer
        summary = f.read(summary_len)

    schemas = {}      # id -> (name, encoding, data_len, data)
    channels = {}     # id -> (schema_id, topic, msg_encoding, metadata)
    stats = None
    chunk_compressions = {}  # compression -> [count, comp_size, uncomp_size]
    op_counts = {}
    attachments = []
    metadatas = []

    for op, content in read_records(summary):
        op_counts[OP.get(op, hex(op))] = op_counts.get(OP.get(op, hex(op)), 0) + 1
        r = Reader(content)
        if op == 0x03:  # Schema
            sid = r.u16(); name = r.s(); enc = r.s(); data = r.by()
            schemas[sid] = (name, enc, len(data), data)
        elif op == 0x04:  # Channel
            cid = r.u16(); sid = r.u16(); topic = r.s(); menc = r.s()
            meta_raw = r.by()
            md = {}
            mr = Reader(meta_raw)
            try:
                while mr.i < len(meta_raw):
                    k = mr.s(); v = mr.s(); md[k] = v
            except Exception:
                pass
            channels[cid] = (sid, topic, menc, md)
        elif op == 0x0B:  # Statistics
            st = {}
            st["message_count"] = r.u64()
            st["schema_count"] = r.u16()
            st["channel_count"] = r.u32()
            st["attachment_count"] = r.u32()
            st["metadata_count"] = r.u32()
            st["chunk_count"] = r.u32()
            st["message_start_time"] = r.u64()
            st["message_end_time"] = r.u64()
            cmc_raw = r.by()
            cmc = {}
            cr = Reader(cmc_raw)
            while cr.i + 10 <= len(cmc_raw):
                ch = cr.u16(); cnt = cr.u64(); cmc[ch] = cnt
            st["channel_message_counts"] = cmc
            stats = st
        elif op == 0x08:  # ChunkIndex
            # message_start, message_end, chunk_start_offset, chunk_length,
            # message_index_offsets (map u16->u64), message_index_length,
            # compression(str), compressed_size(u64), uncompressed_size(u64)
            try:
                _ms = r.u64(); _me = r.u64(); _cso = r.u64(); _cl = r.u64()
                mio = r.by()  # message index offsets map
                _mil = r.u64()
                comp = r.s()
                csize = r.u64(); usize = r.u64()
                e = chunk_compressions.setdefault(comp or "<none>", [0, 0, 0])
                e[0] += 1; e[1] += csize; e[2] += usize
            except Exception:
                pass
        elif op == 0x0A:  # AttachmentIndex
            try:
                _off = r.u64(); _len = r.u64()
                _lt = r.u64(); _ct = r.u64(); _ds = r.u64()
                nm = r.s(); mt = r.s()
                attachments.append((nm, mt, _ds))
            except Exception:
                pass
        elif op == 0x0D:  # MetadataIndex
            try:
                _off = r.u64(); _len = r.u64(); nm = r.s()
                metadatas.append(nm)
            except Exception:
                pass

    # ---------- report ----------
    def human(n):
        for u in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024:
                return f"{n:.1f} {u}"
            n /= 1024
        return f"{n:.1f} PB"

    print("=" * 78)
    print(f"FILE: {path}")
    print(f"Size on disk : {human(size)} ({size:,} bytes)")
    print(f"Profile      : {profile!r}")
    print(f"Library      : {library!r}")
    print("=" * 78)

    if stats:
        t0 = stats["message_start_time"]
        t1 = stats["message_end_time"]
        dur = (t1 - t0) / 1e9
        import datetime
        def ts(ns):
            return datetime.datetime.utcfromtimestamp(ns / 1e9).strftime("%Y-%m-%d %H:%M:%S.%f UTC")
        print("RECORDING SUMMARY")
        print(f"  Total messages : {stats['message_count']:,}")
        print(f"  Channels       : {stats['channel_count']}")
        print(f"  Schemas        : {stats['schema_count']}")
        print(f"  Chunks         : {stats['chunk_count']:,}")
        print(f"  Attachments    : {stats['attachment_count']}")
        print(f"  Metadata recs  : {stats['metadata_count']}")
        print(f"  Start time     : {ts(t0)}")
        print(f"  End time       : {ts(t1)}")
        print(f"  Duration       : {dur:.2f} s  ({dur/60:.2f} min)")
        print()

    if chunk_compressions:
        print("COMPRESSION (from chunk index)")
        for comp, (cnt, cs, us) in chunk_compressions.items():
            ratio = (us / cs) if cs else 0
            print(f"  {comp:10s}: {cnt:,} chunks | compressed {human(cs)} -> uncompressed {human(us)} ({ratio:.2f}x)")
        print()

    print("CHANNELS / TOPICS")
    print("-" * 78)
    cmc = stats["channel_message_counts"] if stats else {}
    rows = []
    for cid, (sid, topic, menc, md) in channels.items():
        sname, senc = (schemas.get(sid, ("<none>", "", 0, b""))[0],
                       schemas.get(sid, ("", "", 0, b""))[1])
        cnt = cmc.get(cid, 0)
        rows.append((cnt, topic, menc, sname, senc, sid, md))
    rows.sort(key=lambda x: -x[0])
    total = stats["message_count"] if stats else sum(r[0] for r in rows)
    dur = ((stats["message_end_time"] - stats["message_start_time"]) / 1e9) if stats else 0
    for cnt, topic, menc, sname, senc, sid, md in rows:
        hz = (cnt / dur) if dur else 0
        print(f"  {topic}")
        print(f"      messages : {cnt:,}  (~{hz:.1f} Hz)" if dur else f"      messages : {cnt:,}")
        print(f"      schema   : {sname}  [{senc or 'n/a'}]  (msg encoding: {menc})")
        if md:
            for k, v in md.items():
                vv = v if len(v) < 120 else v[:117] + "..."
                print(f"      meta[{k}] = {vv}")
    print()

    print("SCHEMAS (definitions)")
    print("-" * 78)
    for sid, (name, enc, dlen, data) in sorted(schemas.items()):
        print(f"  [{sid}] {name}   encoding={enc}   ({dlen} bytes)")
    print()

    if attachments:
        print("ATTACHMENTS")
        for nm, mt, ds in attachments:
            print(f"  {nm}  ({mt})  {human(ds)}")
        print()
    if metadatas:
        print("METADATA RECORDS")
        for nm in metadatas:
            print(f"  {nm}")
        print()

    # dump schema bodies for human-readable text schemas
    print("=" * 78)
    print("SCHEMA BODIES (text schemas only)")
    print("=" * 78)
    for sid, (name, enc, dlen, data) in sorted(schemas.items()):
        if enc in ("ros2msg", "jsonschema", "protobuf", "omgidl", "ros1msg", "flatbuffer"):
            print(f"\n----- [{sid}] {name}  ({enc}) -----")
            if enc in ("ros2msg", "jsonschema", "omgidl", "ros1msg"):
                print(data.decode("utf-8", "replace"))
            else:
                print(f"  (binary {enc} schema, {dlen} bytes — not text)")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "20260615_180456_T4_bulk.mcap"
    main(p)
