"""Read thermal-camera frames out of the Agility Forge HMR mcap logs.

Why this module exists
----------------------
The repo's ``sample_mcap.py`` **silently extracts nothing** from these frames. It
assumes the stock ``foxglove.RawImage`` layout, and this stream does not use it.
Determined empirically from the bytes of `20260717_135009.mcap`:

===========  ==========================  ==================================
field no.    stock foxglove.RawImage     THIS stream
===========  ==========================  ==================================
1            timestamp                   timestamp
2            frame_id (string)           **width  (fixed32)**
3            width  (fixed32)            **height (fixed32)**
4            height (fixed32)            **encoding (string)**
5            encoding (string)           **step (fixed32)**
6            step (fixed32)              **data (bytes)**
7            data (bytes)                --
===========  ==========================  ==================================

There is no ``frame_id`` and everything after it shifts down one. Compounding it,
``width``/``height``/``step`` are **fixed32** (wire type 5), not varint (wire type 0)
— a parser that reads them as varint skips them entirely and returns an empty dict
rather than failing loudly.

Verified self-consistent on the 07-17 file: width 382, step 764 = 2*382, data
220032 = 382*288*2, encoding "16UC1".

Units
-----
Colton Wright, 2026-07-17: *"Units of the thermal camera matrix are deci-kelvin."*
So ``kelvin = raw_uint16 / 10``.

⚠️ Those are **radiometric** temperatures, i.e. they depend on the emissivity the
camera was configured with. 316L's emissivity roughly doubles as it oxidises, so a
fixed camera setting drifts against the true surface. See
``docs/AS_BUILT_AGILITY_FORGE.md`` — this is the dominant uncertainty in any
calibration against these numbers, not a detail.

Geometry
--------
In the 07-17 framing the **rod runs vertically** (rows = axial) and the coil turns
appear as *cool* bars crossing it, because the water-cooled copper occludes the hot
rod behind it. Bright regions are rod glimpsed between turns.
"""

from __future__ import annotations

import struct
import subprocess
from dataclasses import dataclass

import numpy as np

MAGIC = b"\x89MCAP0\r\n"
THERMAL_TOPIC = "hmr/sensors/thermalcam"
DECI_KELVIN = 10.0

#: Usable heating window for `20260717_135009.mcap`, seconds from the first message.
#: Colton said "380s into the mcap to the end". The START is his and is right; "the
#: end" is NOT. Reading every frame at full rate shows p99 surface temperature
#: falling 925 -> 400 C between 568.4 s and 570.4 s, then flat at ~400 C until the
#: file ends at 573.3 s.
#:
#: That collapse is ~200 K/s. Radiative cooling of a 38.1 mm 316L rod at 925 C is
#: only ~1 K/s — eps*sigma*(T^4 - T_inf^4) * (2/r) / (rho*Cp) — so it is ~200x too
#: fast to be the rod cooling in place. The rod leaves the field of view and the
#: flat tail is background, not metal. Fitting to "the end" fits the withdrawal.
#:
#: NOTE this is emphatically NOT the power-off cooling curve worth requesting from
#: Colton: far too fast to carry loss information. Worth asking what happens here.
HEAT_START_S = 380.0
HEAT_END_S = 568.4


def _decompress_zstd(data: bytes, size_hint: int) -> bytes:
    """zstandard if installed, else shell out to the zstd CLI."""
    try:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(data, max_output_size=size_hint + 4096)
    except ImportError:
        pass
    p = subprocess.run(["zstd", "-d", "-c"], input=data,
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(
            "chunk is zstd-compressed but neither the `zstandard` package nor the "
            "`zstd` CLI is available"
        )
    return p.stdout


class _Reader:
    __slots__ = ("b", "i")

    def __init__(self, b: bytes):
        self.b = b
        self.i = 0

    def u16(self):
        v = struct.unpack_from("<H", self.b, self.i)[0]; self.i += 2; return v

    def u32(self):
        v = struct.unpack_from("<I", self.b, self.i)[0]; self.i += 4; return v

    def u64(self):
        v = struct.unpack_from("<Q", self.b, self.i)[0]; self.i += 8; return v

    def string(self):
        n = self.u32()
        v = self.b[self.i:self.i + n].decode("utf-8", "replace"); self.i += n; return v


def _records(buf: bytes):
    """Yield (opcode, payload) for top-level MCAP records."""
    i, n = 0, len(buf)
    while i + 9 <= n:
        op = buf[i]
        length = struct.unpack_from("<Q", buf, i + 1)[0]
        start = i + 9
        end = start + length
        if end > n:
            break
        yield op, buf[start:end]
        i = end


def parse_raw_image(payload: bytes) -> dict:
    """Decode this stream's RawImage variant. See the module docstring."""
    out: dict = {}
    i, n = 0, len(payload)

    def varint(i):
        shift = val = 0
        while True:
            b = payload[i]; i += 1
            val |= (b & 0x7F) << shift
            if not (b & 0x80):
                return val, i
            shift += 7

    while i < n:
        key, i = varint(i)
        field, wire = key >> 3, key & 7
        if wire == 2:                                  # length-delimited
            ln, i = varint(i)
            chunk = payload[i:i + ln]; i += ln
            if field == 4:
                out["encoding"] = chunk.decode("utf-8", "replace")
            elif field == 6:
                out["data"] = chunk
        elif wire == 5:                                # fixed32
            v = struct.unpack_from("<I", payload, i)[0]; i += 4
            if field == 2:
                out["width"] = v
            elif field == 3:
                out["height"] = v
            elif field == 5:
                out["step"] = v
        elif wire == 0:
            _v, i = varint(i)
        elif wire == 1:
            i += 8
        else:
            break
    return out


@dataclass
class ThermalFrame:
    t_s: float           # seconds since first message in the file
    log_time_ns: int
    kelvin: np.ndarray   # [height, width]

    @property
    def celsius(self) -> np.ndarray:
        return self.kelvin - 273.15


class ForgeMcap:
    """Random-access reader over the summary/chunk index of an HMR mcap."""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        size = self._f.seek(0, 2)
        self._f.seek(size - 8 - 29)
        tail = self._f.read(8 + 29)
        if tail[-8:] != MAGIC:
            raise ValueError(f"{path}: bad end magic — not an MCAP file, or truncated")
        summary_start, _soff, _crc = struct.unpack_from("<QQI", tail[:29], 9)
        if summary_start == 0:
            raise ValueError(f"{path}: no summary section; cannot index without a full scan")
        self._f.seek(summary_start)
        summary = self._f.read((size - 8 - 29) - summary_start)

        self.channels: dict[int, str] = {}
        self.chunks: list[tuple[int, int, int, int]] = []
        self.t0_ns = self.t1_ns = 0
        for op, payload in _records(summary):
            r = _Reader(payload)
            if op == 0x04:                              # Channel
                cid = r.u16(); r.u16(); topic = r.string(); r.string()
                self.channels[cid] = topic
            elif op == 0x08:                            # ChunkIndex
                ms = r.u64(); me = r.u64(); off = r.u64(); ln = r.u64()
                self.chunks.append((ms, me, off, ln))
            elif op == 0x0B:                            # Statistics
                r.u64(); r.u16(); r.u32(); r.u32(); r.u32(); r.u32()
                self.t0_ns = r.u64(); self.t1_ns = r.u64()
        self.chunks.sort()

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def duration_s(self) -> float:
        return (self.t1_ns - self.t0_ns) / 1e9

    def _channel_id(self, topic: str) -> int:
        for cid, name in self.channels.items():
            if name == topic:
                return cid
        raise KeyError(f"topic {topic!r} not in {sorted(self.channels.values())}")

    def _load_chunk(self, offset: int, length: int) -> bytes:
        self._f.seek(offset)
        rec = self._f.read(length)
        r = _Reader(rec[9:])
        r.u64(); r.u64(); uncompressed = r.u64(); r.u32()
        compression = r.string(); reclen = r.u64()
        body = r.b[r.i:r.i + reclen]
        return _decompress_zstd(body, uncompressed) if compression == "zstd" else body

    def thermal_frames(self, start_s: float = 0.0, end_s: float | None = None,
                       chunk_stride: int = 1, frames_per_chunk: int | None = None):
        """Yield :class:`ThermalFrame` between ``start_s`` and ``end_s``.

        ``chunk_stride`` subsamples chunks (each holds a couple of seconds); use it
        to sweep a long file cheaply. ``frames_per_chunk`` caps frames taken from
        each decompressed chunk — 1 gives an evenly spaced sample.
        """
        cid = self._channel_id(THERMAL_TOPIC)
        lo = self.t0_ns + int(start_s * 1e9)
        hi = self.t1_ns if end_s is None else self.t0_ns + int(end_s * 1e9)
        selected = [c for c in self.chunks if c[1] >= lo and c[0] <= hi]
        for ms, me, off, ln in selected[::chunk_stride]:
            taken = 0
            for op, payload in _records(self._load_chunk(off, ln)):
                if op != 0x05 or struct.unpack_from("<H", payload, 0)[0] != cid:
                    continue
                log_time = struct.unpack_from("<Q", payload, 6)[0]
                if not (lo <= log_time <= hi):
                    continue
                info = parse_raw_image(payload[22:])
                w, h, data = info.get("width"), info.get("height"), info.get("data")
                if not (w and h and data and len(data) >= w * h * 2):
                    continue
                raw = np.frombuffer(data[:w * h * 2], dtype="<u2").reshape(h, w)
                yield ThermalFrame(
                    t_s=(log_time - self.t0_ns) / 1e9,
                    log_time_ns=log_time,
                    kelvin=raw.astype(np.float64) / DECI_KELVIN,
                )
                taken += 1
                if frames_per_chunk is not None and taken >= frames_per_chunk:
                    break

    def temperature_series(self, start_s: float = 0.0, end_s: float | None = None,
                           chunk_stride: int = 1) -> np.ndarray:
        """One row per sampled frame: ``[t_s, max, p99, p95, median, min]`` in Celsius.

        p99 is the better surface proxy — ``max`` is a single-pixel statistic and
        visibly noisier frame to frame.
        """
        rows = []
        for fr in self.thermal_frames(start_s, end_s, chunk_stride, frames_per_chunk=1):
            c = fr.celsius
            rows.append([fr.t_s, c.max(), np.percentile(c, 99),
                         np.percentile(c, 95), np.median(c), c.min()])
        return np.array(rows)


def pixels_per_mm(frame: ThermalFrame, rod_diameter_mm: float = 38.1) -> float:
    """Self-calibrate the image scale from the rod's known diameter.

    The rod runs vertically, so its diameter is measured ACROSS columns. Takes the
    half-height width of the hot band. Returns px/mm.

    ⚠️ **This is threshold-sensitive — treat the answer as +/- 25%, not precise.**
    The rod edges bloom thermally, so where you place the edge moves the result a
    lot. On the 07-17 file this half-height criterion gives ~4.8 px/mm, while
    reading the edges by eye off a rendered frame gives ~3.7-4.0, against the
    3.8791 Colton quoted for the *June* run. Those do not agree, and the frame-width
    cross-check does not discriminate: 3.879 makes the frame 98x74 mm and 4.803
    makes it 80x60 mm, and the coil (76 mm) is "mostly in view" either way.

    Resolving this properly needs Colton's pixel-mapping code (offered 2026-06-29,
    never taken up), or a frame with a known reference edge in it.
    """
    c = frame.celsius
    colmax = c.max(axis=0)
    half = 0.5 * (np.percentile(colmax, 5) + np.percentile(colmax, 95))
    above = np.where(colmax > half)[0]
    if len(above) < 2:
        raise ValueError("could not locate the rod: no clear hot band across columns")
    return (above.max() - above.min() + 1) / rod_diameter_mm
