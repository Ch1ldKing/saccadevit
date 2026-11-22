# SaccadeViT

**Look · keep · stop on one ~5.7M-parameter ViT.** A coarse look proposes where to aim. A task-conditioned policy picks a crop, a FocusViT-style glimpse keeps the top-K translated patches, and a GRU accumulates evidence until the prediction is confident. The speed-up is measured in p50/p95 milliseconds, not in FLOPs.

Method and rationale: [`../DESIGN.md`](../DESIGN.md). Project slides: `../../Saccadic_ViT_Presentation_EN 1.pptx`.

## Setup

```bash
uv sync                      # Python 3.11, torch, timm
bash scripts/get_data.sh     # CUB-200-2011 -> data/CUB_200_2011 (CIFAR-100 downloads itself)
uv run pytest -q             # ~1 min on CPU, fake data
```

Run every command from the repo root. Outputs go to `runs/<exp>/seed<k>/`.

## Experiments

There is one YAML file per experiment in `configs/exp/`. Each inherits from `configs/base.yaml` through `_base_`. A run is named after its path, so `canvas/b0_full.yaml` becomes `canvas-b0_full`.

| Config | Command | What it answers |
|---|---|---|
| `b0_full` | `train` | B0 upper bound. Also the init for every other run (gate M1) |
| `stage_a` | `train` | Trains M3/M4 on teacher fixations (gate M2) |
| `b1_random` / `b2_saliency` / `b3_focusvit` / `oracle_gt` | `eval` | Other aims, same Stage-A weights and same budget |
| `b4_merge` | `train` | Merging vs. gathering |
| `b5_ours` | `train`, `eval` | REINFORCE-trained task-conditioned policy (H1, gate M3) |
| `a1_no_offset_stage_a` → `a1_no_offset` | `train` ×2 | Offset term removed |
| `a2_fixed_k` | `eval` | No stopping (H2, at matched accuracy) |
| `a3_keep_sweep` | `sweep` | One checkpoint, many budgets (H3) |
| `a4_coarse_only` | `eval` | What the looks buy at all |
| `a5_no_task` | `train`, `eval` | Task signal off (H1) |
| `canvas/*`, `cifar/*` | same | Second task / control |

```bash
uv run sv train configs/exp/b0_full.yaml seed=1          # any key=value override
uv run sv eval  configs/exp/b5_ours.yaml --split test
uv run sv sweep configs/exp/a3_keep_sweep.yaml
uv run sv latency configs/exp/b5_ours.yaml eval.tau=0.4 eval.device=cuda
uv run sv compare 'runs/b2_saliency/seed*/eval_test.npz' 'runs/b5_ours/seed*/eval_test.npz' --k 2
bash scripts/run_cub.sh                                    # the full CUB matrix, 3 seeds
```

## Layout

```
src/saccadevit/
  config.py           YAML + _base_ inheritance + dotlist overrides
  data/               cub (boxes, 15 parts), canvas (synthetic 1024² search), cifar (control)
  models/
    backbone.py       shared timm ViT wrapper (patchify / encode) + B0
    coarse.py         M1 coarse look
    policy.py         M2 learned policy, task encoder q, random/saliency/center/gt policies
    glimpse.py        M3 crop, patch translator, offset/location PE, perturbed top-K
    aggregator.py     M4 GRU, entropy stop rule
    saccade.py        the loop: forward (K looks, batched) and infer (batch 1, real stopping)
    merging.py        B4 ATME-style merging
  engine/             losses (stage full / a / b), train, evaluate, sweep, latency, stats
```
