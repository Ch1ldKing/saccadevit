# Experiment log

Hand-written record of runs, diagnostics and the decisions they caused. Final numbers live in the auto-generated [CURRENT_RESULTS.md](CURRENT_RESULTS.md).

Server: `hitcvsz4`, `/data1/yhy/projects/saccadevit`, RTX 3090 GPUs 1 and 3 (0 and 2 belong to other users). Logs: `output/logs/gpu{1,3}.log`. Diagnostic runs: `output/diag/<name>/`.

## 2026-11-23: setup

| Step | Outcome |
|---|---|
| `uv sync` | PyPI ran at 0.86 MB/s, so installed via the Tsinghua mirror (`UV_DEFAULT_INDEX`). Same versions: torch 2.15+cu132, timm 1.0.31 |
| Weights | HF Hub blocked, so fetched via `HF_ENDPOINT=https://hf-mirror.com` |
| CUB download | Caltech returns 403 for the default urllib agent, so the client now identifies itself as `saccadevit-setup/0.1` |
| `third_party/ptopk` | The server's git 2.34 cannot fetch the google-research repo (HTTP/2 framing, ref listing), so the committed snapshot is used. Not imported at runtime |
| Canvas | 1024² made data loading the bottleneck (1.35 it/s, GPU idle), so switched to 512² with 32–56 px targets (same coarse-view geometry): 4.3 it/s |

## 2026-11-23: preliminary CUB seed 0 (v0 Stage A) exposed a selection failure

| Run (v0 defaults) | Test acc % |
|---|---|
| B0 full image | **76.06** (val 77.30) |
| B1 random, 3 looks, ρ=0.15 | 13.15 |
| B2 saliency | 17.03 |
| B3 one whole-image look, ρ=0.3 | 26.80 |
| Oracle gt, 3 looks | 25.20 |

The v0 Stage A model (val, center view, 1 look) shows where it breaks:

| Tokens kept | Scorer top-K | Random K | Scorer bottom-K |
|---|---|---|---|
| 100% | 68.8 | | |
| 15% | 13.5 | 20.5 | 20.7 |

**Finding 1.** The learned scorer selects *worse than random*. The perturbed-top-K gradient (σ=0.05 on z-scored scores, 16 draws) almost never flips a selection, so it carries no signal.
**Finding 2.** With only the last 2 blocks tuned, even random sparse token sets lose ~50 points.

### Short Stage-A variants (8 epochs, best val %, random looks, ρ=0.15)

| Variant | Change | Val % |
|---|---|---|
| d1 | σ 0.5, 32 draws | 12.2 |
| **d2** | tune blocks 3–12 (coarse blocks 1–2 frozen) | **24.9** |
| d3 | d1 + d2 | 20.9 |
| d4 | d3 + ρ ∈ [0.1, 0.5] | 16.4 |

Probe of d2 (val, 1 look unless stated):

| Policy | ρ=1.0 | ρ=0.5 | ρ=0.3 | ρ=0.15 (scorer / random-K) |
|---|---|---|---|---|
| center | 68.3 | 55.8 / 52.6 | 46.1 / 39.7 | 29.5 / 24.7 |
| gt | 71.3 | 59.8 / 61.4 | 50.3 / 52.8 | 36.2 / 33.2 |
| gt, 3 looks | 71.6 | 63.3 / 67.3 | 54.3 / 56.9 | 39.7 / 37.7 |

So backbone tuning is the main lever, but the scorer is still no better than random. Next: supervise the scorer with the backbone's own CLS attention (`model.select_distill`) and fold the dropped tokens into one EViT rest token (`model.fuse_rest`). Variants e1–e3 are running.
