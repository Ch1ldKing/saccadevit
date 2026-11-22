# Runbook

Everything goes through `make`. Run from the repo root on the GPU host (one L40 48 GB).

## Setup (once per machine)

```bash
make setup          # uv sync; CUB + CIFAR-100 -> input/datasets; ViT weights -> input/weights; third_party/ snapshots
make input_verify   # file counts + SHA-256 (recorded in input/MANIFEST.json at fetch time)
make validate_configurations
make test
```

- `input/` and `output/` are git-ignored except `.gitkeep`. `third_party/` is committed. `make third_party_fetch` re-snapshots it at the commits in `configs/assets.yaml`.
- Weights load from `input/weights/...` (`model.weights`), so training never touches the HF hub.
- Do not fetch CUB inside the Obsidian vault copy of this repo (11.8k JPEGs would be indexed). Use the GPU host.

## Experiments

```bash
make help                                  # every target
make -n preliminary                        # preview the commands
make canvas                                # pipeline check: the learned aim must beat random here
make preliminary                           # seed 0: B0, Stage A, B1-B3 + oracle, B5, sweep, latency
make cub SEED=1                            # full CUB matrix for one seed
make cub_3_seeds                           # seeds 0-2, latency, H1/H2 comparisons
make train_b5_ours SEED=0 ARGS="train.epochs=5 train.cost_weight=0.1"   # any config override
```

Dependencies: `train_b0_full` → `train_stage_a` → (`evaluate_b1/b2/b3/oracle`, `train_b5_ours`, `train_a5_no_task`) → `evaluate_*`, `sweep_a3_keep_sweep`. B4 and A1 also start from `train_b0_full`.

## Outputs (`output/<exp>/seed<k>/`)

| File | From |
|---|---|
| `config.yaml`, `metrics.jsonl`, `best.pt`, `last.pt` | train |
| `eval_test.npz`, `summary_test.json` (acc@k, the τ table) | evaluate |
| `sweep.csv` (`rule == tau*` rows are the headline points) | sweep |
| `latency_{cuda,cpu}.json` (p50, p95, per-module ms, GFLOPs, peak memory) | latency |

After the sweep, pass its τ* to `latency_gpu_b5_ours_stop TAU=…` and `compare_h2_stop TAU=…`.
