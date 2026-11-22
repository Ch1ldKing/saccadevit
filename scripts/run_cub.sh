#!/usr/bin/env bash
# Full CUB matrix in dependency order. SEEDS="0 1 2" by default.
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}
E=configs/exp
for s in $SEEDS; do
  uv run sv train $E/b0_full.yaml seed=$s
  uv run sv train $E/stage_a.yaml seed=$s
  uv run sv train $E/b4_merge.yaml seed=$s
  uv run sv train $E/b5_ours.yaml seed=$s
  uv run sv train $E/a5_no_task.yaml seed=$s
  uv run sv train $E/a1_no_offset_stage_a.yaml seed=$s
  uv run sv train $E/a1_no_offset.yaml seed=$s
  for x in b0_full b4_merge b1_random b2_saliency b3_focusvit oracle_gt b5_ours a1_no_offset a2_fixed_k a4_coarse_only a5_no_task; do
    uv run sv eval $E/$x.yaml seed=$s
  done
  uv run sv sweep $E/a3_keep_sweep.yaml seed=$s
done
for x in b0_full b3_focusvit b4_merge b5_ours; do
  uv run sv latency $E/$x.yaml eval.device=cuda
  uv run sv latency $E/$x.yaml eval.device=cpu eval.iters=200
done
uv run sv latency $E/b5_ours.yaml eval.device=cuda eval.tau=0.4
