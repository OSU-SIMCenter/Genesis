"""One-time: extract the real per-hit billet meshes from the 319 MB .pt into compact .npz.

WHY THIS EXISTS. The metric script needs ~5 MB of mesh per hit, but reaching it via
torch.load pulls in torch (and with it a CUDA runtime that reserves multiple GB of virtual
address space on import). On this machine -- WSL capped near 7 GB, host at ~1 GB free and
36.5/41.5 GB committed -- that is the difference between a script that runs and one that
contributes to killing the VM. It also made an RLIMIT_AS guard useless, because the cap
counts reserved VA rather than resident memory.

Extracting once decouples the geometry metric from torch entirely: afterwards it needs only
numpy + scipy + trimesh, and the memory guard becomes meaningful again.

Run this ONCE. It is the only step that needs torch, and it is idempotent (skips hits
already written).
"""
import os
import pathlib
import sys

import numpy as np

OUT = pathlib.Path.home() / "GitHub/Genesis/forge_common/main/outputs/real_meshes"
PT = (pathlib.Path.home()
      / "GitHub/Genesis/forge_common/models/forge-net/forge_net/data/datasets/agf_data/2026-06-29.pt")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [h for h in range(1, 18) if not (OUT / ("hit_%02d.npz" % h)).exists()]
    if not todo:
        print("all 17 hits already extracted ->", OUT)
        return 0

    import torch  # imported only here, only for this one-time step
    d = torch.load(PT, map_location="cpu", weights_only=False)

    # Per-hit force and the tool ids travel with the meshes so downstream never needs the
    # .pt again. anvil_tool/hammer_tool are both 2 for all 17 hits -- one tool geometry.
    meta = dict(F=d["F"].numpy(), anvil_tool=d["anvil_tool"].numpy(),
                hammer_tool=d["hammer_tool"].numpy())
    np.savez_compressed(OUT / "meta.npz", **meta)

    for h in todo:
        i = h - 1
        out = {}
        for side, vk, tk, nvk, ntk in (
            ("before", "vertices_k", "triangles_k", "n_vertices_k", "n_triangles_k"),
            ("after", "vertices_k1", "triangles_k1", "n_vertices_k1", "n_triangles_k1"),
        ):
            nv, nt = int(d[nvk][i]), int(d[ntk][i])
            out["V_" + side] = np.asarray(d[vk][i, :nv], dtype=np.float32)
            # int32 is ample: max vertex index is ~145k, far under 2^31.
            out["T_" + side] = np.asarray(d[tk][i, :nt], dtype=np.int32)
        p = OUT / ("hit_%02d.npz" % h)
        np.savez_compressed(p, **out)
        print("hit %2d  V_after=%-7d T_after=%-7d  ->  %s (%.1f MB)"
              % (h, len(out["V_after"]), len(out["T_after"]), p.name,
                 p.stat().st_size / 1e6))
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
