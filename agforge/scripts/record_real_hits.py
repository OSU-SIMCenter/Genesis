"""Record the real Agility Forge hit sequence via forge_common's genesis adapter.

Lives in Genesis so forge_common stays untouched: after ``init_stock``, this
script arms ``AgForgeRecorder`` itself, runs the real hits, then flushes with
stock geometry attrs for ``python -m agforge.replay_episode``.

Requires forge_common on PYTHONPATH (the adapter + real-hit loaders).

Usage (from Genesis workspace root):

    LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH \\
    PYTHONPATH=forge_common/main \\
    pixi run --manifest-path aims-genesis/nsf-demo/pixi.toml python -u \\
      -m agforge.scripts.record_real_hits \\
      --n-hits 9 --show-viewer \\
      --record-out forge_common/main/outputs/genesis_real_n9.h5

Loop replay (cwd = aims-genesis/nsf-demo):

    pixi run python -m agforge.replay_episode \\
      --data /abs/path/to/genesis_real_n9.h5 -e ep_000000 --loop
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py

# Optional: discover a sibling forge_common/main if PYTHONPATH was not set.
_HERE = Path(__file__).resolve()
# .../Genesis/aims-genesis/nsf-demo/agforge/scripts/this.py -> parents[4] = workspace
_CANDIDATES = [
    _HERE.parents[4] / "forge_common" / "main",
    _HERE.parents[3] / "forge_common" / "main",
]
for _p in _CANDIDATES:
    if (_p / "forge_common").is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
        break

try:
    from forge_common.adapters.genesis_forge_adapter import GenesisForgeAdapter
    from forge_common.policies import run_hit_sequence
    from forge_common.press_tool import die_contact_axial_width_mm
    from forge_common.real_data import REAL_DATA_PT, load_real_hits_for_sim
    from forge_common.real_scale import REAL_STOCK_LENGTH_MM, REAL_STOCK_RADIUS_MM
except ImportError as e:
    print(
        "ERROR: forge_common is required. From the Genesis workspace root, set\n"
        "  PYTHONPATH=forge_common/main\n"
        f"Import failed: {e}",
        file=sys.stderr,
    )
    sys.exit(1)


def _export_named_episode(
    src_shard: str, src_ep: str, dest_path: str, dest_ep: str = "ep_000000"
):
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_shard, "r") as src, h5py.File(dest, "w") as dst:
        if src_ep not in src:
            raise KeyError(f"episode {src_ep!r} not in {src_shard}")
        src.copy(src_ep, dst, name=dest_ep)
    return str(dest.resolve()), dest_ep


def _flush_episode(state, *, success: bool, language_instruction: str):
    """Flush AgForgeRecorder with stock attrs for replay_episode matching."""
    rec = state.controller.recorder
    if not rec.is_recording or len(rec.buffer["qpos"]) == 0:
        return None
    shard_id = rec.global_episode_id // rec.shard_capacity
    ep_name = f"ep_{rec.global_episode_id:06d}"
    shard_path = os.path.join(rec.train_dir, f"shard_{shard_id:04d}.h5")
    rec.flush_episode(
        success_flag=success,
        language_instruction=language_instruction,
        extra_attrs={
            "stock_diameter_m": 2.0 * float(state.radius_mm) / 1000.0,
            "stock_length_m": float(state.length_mm) / 1000.0,
            "gripper_axial_width_m": float(state.press_width_mm) / 1000.0,
            "n_particles": int(state.env.mpm_entity.n_particles),
        },
    )
    print(f"[record_real_hits] flushed {shard_path} / {ep_name}")
    return shard_path, ep_name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-hits",
        type=int,
        default=9,
        help="how many of the 17 real hits to replay (default 9; pass 17 for full)",
    )
    p.add_argument(
        "--show-viewer",
        action="store_true",
        help="open the live Genesis viewer while hits play (needs DISPLAY)",
    )
    p.add_argument(
        "--record-out",
        default=None,
        help="copy flushed episode into this standalone .h5 as ep_000000",
    )
    args = p.parse_args()

    hits = load_real_hits_for_sim("genesis", args.n_hits)
    print(f"Recording from {REAL_DATA_PT.name}: {len(hits)} real hit(s)")
    for i, h in enumerate(hits):
        print(f"  {i + 1}: rho={h.rho:.3f}mm, phi={h.phi:.4f}rad, z={h.z:.3f}mm")

    # forge_common adapter intentionally leaves the recorder OFF; arm it here.
    adapter = GenesisForgeAdapter(
        press_width_mm=die_contact_axial_width_mm(),
        show_viewer=args.show_viewer,
    )
    state = adapter.init_stock(
        radius_mm=REAL_STOCK_RADIUS_MM, length_mm=REAL_STOCK_LENGTH_MM
    )
    state.controller.recorder.is_recording = True
    print(f"[record_real_hits] recording ON -> {state.controller.recorder.train_dir}")

    hit_i = [0]

    def on_hit(state, hit, mesh):
        hit_i[0] += 1
        print(
            f"  hit {hit_i[0]}/{len(hits)}: rho={hit.rho:.3f}mm, "
            f"phi={hit.phi:.4f}rad, z={hit.z:.3f}mm"
        )

    t0 = time.time()
    result = run_hit_sequence(adapter, state, hits, on_hit=on_hit)
    elapsed = time.time() - t0

    if result.succeeded:
        print(
            f"genesis: all {result.hits_completed}/{result.total_hits_planned} "
            f"real hits in {elapsed:.1f}s"
        )
    else:
        print(
            f"genesis: STOPPED after {result.hits_completed}/"
            f"{result.total_hits_planned} real hits ({elapsed:.1f}s) -- "
            f"{type(result.error).__name__}: {result.error}"
        )

    episode = _flush_episode(
        state,
        success=result.succeeded,
        language_instruction=(
            f"Real Agility Forge hit-sequence replay "
            f"({result.hits_completed}/{result.total_hits_planned} hits)"
        ),
    )
    if episode is None:
        print("No episode flushed (empty recorder buffer).")
        sys.exit(1)

    shard_path, ep_name = episode
    replay_data, replay_ep = shard_path, ep_name
    if args.record_out:
        replay_data, replay_ep = _export_named_episode(
            shard_path, ep_name, args.record_out
        )
        print(f"Named export: {replay_data} ({replay_ep})")

    print(
        "Loop replay:\n"
        f"  pixi run python -m agforge.replay_episode "
        f"--data {replay_data} -e {replay_ep} --loop"
    )
    if not result.succeeded:
        print("Note: hit sequence did not finish; episode was still written.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
