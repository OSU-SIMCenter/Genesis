#!/bin/bash
# D1a re-run with the die-balance controller SUPPRESSED.
#
# The archived batch_speed_* sweep ran with an active stall whose observed exposure grows 10x
# as the press slows (1.3 -> 13.7% of frames), the same artifact that made workstream A's
# KE/IE table read backwards. This re-runs the ends of the speed ladder with the controller
# out, so geometry spread can be attributed to the similarity transform rather than the loop.
#
#   AGF_FORCE_BALANCE_GAIN=1.5e-5        raises |dF|_stall
#   AGF_FORCE_IMBALANCE_THRESHOLD=1e12   removes feed-rate MODULATION as well as stall
#   AGF_MAX_FORCE=1e9                    removes the truncation stop
#
# Matched to the archived batches: cpd 10, approach CFL 0.05, ppc 2.0, and the SAME real-scan
# billet IC (run_meta of batch_speed_25p0 records billet_hit01_before_d8000.obj -- omitting it
# would silently swap in a cylinder IC and compare against a different initial condition).
# AGF_CONTACT_RUNTIME_SWITCH=1 is required by the driver: without it the contact mode is baked
# into the kernels at scene build and every arm after the first runs the first arm's config.
#
# The "####" banner is emitted so stall_audit.py parses this log unchanged -- that is how
# suppression gets VERIFIED rather than assumed.
set -u
cd /home/timothy/GitHub/Genesis || exit 1
MAN=aims-genesis/nsf-demo/pixi.toml
BA=aims-genesis/nsf-demo/agforge/analysis/batch_arms.py
OUTROOT=/home/timothy/GitHub/Genesis/forge_common/main/outputs
NHITS="${NHITS:-17}"
PREFIX="${PREFIX:-batch_d1aclean_}"

export PYTHONPATH=/home/timothy/GitHub/Genesis/forge_common/main
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
export AGF_CONTACT_RUNTIME_SWITCH=1
export AGF_ENABLE_CPIC=0
export AGF_BILLET_MESH=/home/timothy/GitHub/Genesis/forge_common/main/outputs/real_meshes/billet_hit01_before_d8000.obj
export AGF_FORCE_BALANCE_GAIN=1.5e-5
export AGF_FORCE_IMBALANCE_THRESHOLD=1e12
export AGF_MAX_FORCE=1e9
export AGF_CELLS_PER_DIAMETER=10
export AGF_APPROACH_CFL_RATIO=0.05
export AGF_PPC_DIVISOR=2.0
export AGF_MAX_PARTICLE_VELOCITY=100.0

run_one () {
  SPEED="$1"; TAG="$2"
  echo "################ pressing_speed=${SPEED} m/s per jaw  ($(date +%H:%M:%S))"
  AGF_PRESSING_SPEED="$SPEED" /home/timothy/.pixi/bin/pixi run --manifest-path "$MAN" --frozen \
    python "$BA" --n-hits "$NHITS" --arms g1_grid_prod,p5_penalty --out "${PREFIX}${TAG}"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  ABORT: batch_arms exited $rc at speed $SPEED"
    exit $rc
  fi
  D="${OUTROOT}/${PREFIX}${TAG}"
  if [ ! -d "$D" ]; then
    echo "  ABORT: $D not created"
    exit 1
  fi
  # Provenance sidecar. run_meta.json does NOT record controller state, which is exactly what
  # a later re-score has to infer. Written beside the batch rather than by editing the shared
  # driver, which lives on another workstream's branch.
  cat > "${D}/controller_meta.json" <<JSON
{
  "pressing_speed": ${SPEED},
  "force_balance_gain": ${AGF_FORCE_BALANCE_GAIN},
  "force_imbalance_threshold": ${AGF_FORCE_IMBALANCE_THRESHOLD},
  "max_force": ${AGF_MAX_FORCE},
  "cells_per_diameter": ${AGF_CELLS_PER_DIAMETER},
  "approach_cfl_ratio": ${AGF_APPROACH_CFL_RATIO},
  "billet_mesh": "${AGF_BILLET_MESH}",
  "n_hits": ${NHITS},
  "note": "controller suppressed: modulation AND stall removed; force stop removed",
  "card": "sourced 316L, nsf-demo 6ee71236 -- NEW card; archived batch_speed_* are OLD card"
}
JSON
  echo "  wrote ${D}/controller_meta.json"
}

t0=$(date +%s)
run_one 25.0  25p0
if [ "${ONLY_FAST:-0}" != "1" ]; then
  run_one 3.125 3p125
fi
echo "TOTAL $(( ($(date +%s) - t0) / 60 )) min"
