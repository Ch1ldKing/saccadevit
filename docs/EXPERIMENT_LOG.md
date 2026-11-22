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

So backbone tuning is the main lever, but the scorer is still no better than random.

### Scorer supervision and rest token (8 epochs, on top of d2; probe at ρ=0.15, scorer / random-K)

| Variant | Best val % | center, 1 look | gt, 1 look | gt, 3 looks |
|---|---|---|---|---|
| e1: CLS-attention distillation | 14.7 | 18.9 / 19.2 | 27.2 / 30.4 | 31.7 / 38.9 |
| e2: fused rest token | 20.0 | 24.4 / 24.9 | 31.6 / 37.4 | 34.9 / **46.6** |
| e3: both | 11.7 | 9.0 / 17.9 | 16.9 / 33.7 | 21.0 / 37.7 |

**Finding 3.** Importance-based selection loses to random subsets in every variant, and teacher attention makes it worse. A sparse ViT input needs **coverage**: top-K clusters on salient, redundant patches. The rest token helps by about 5–8 points.

### Coverage-preserving glimpses (8 epochs, on top of d2)

| Variant | Best val % | center ρ=.15 | gt 1 look ρ=.3 | gt 1 look ρ=.15 | gt 3 looks ρ=.15 | random 3 looks ρ=.15 |
|---|---|---|---|---|---|---|
| d2 learned top-K | 24.9 | 29.5 | 50.3 | 36.2 | 39.7 | 23.9 |
| f1 grid + rest token | 28.4 | 35.9 | 64.9 | 39.2 | 49.9 | 26.2 |
| **f2 resize** (crop at g×g, all tokens) | **39.6** | 31.2 | **71.0** | **54.1** | **61.3** | **36.4** |

**Decision (2026-11-23).** The default glimpse becomes `select: resize` with blocks 3–12 tuned, and crop scales {80, 112, 160, 224}. This is a foveal glimpse: a budget of g² tokens covers the whole crop, and zoom is the policy's scale choice. FocusViT-style learned top-K becomes ablation **A6** (`a6_topk_stage_a`), and B3 evaluates that checkpoint so B3 stays faithful to FocusViT. The v0 runs are archived in `output/_v0/`. B0 and canvas B0 are unaffected and kept.


## 2026-11-23: preliminary CUB seed 0 with the resize glimpse

Test %, 25 tokens per look (ρ=0.15). Stage A best val 41.9; A6 (top-K) 26.7; B5 66.9.

| Run | 1 look | 2 looks | 3 looks |
|---|---|---|---|
| B1 random (Stage-A weights) | 23.3 | 33.3 | 40.3 |
| B2 saliency (Stage-A weights) | 38.2 | 45.5 | 50.5 |
| **B5 learned, task-conditioned** | **58.8** | **64.2** | **66.5** |
| Oracle gt (Stage-A weights) | 59.8 | 66.4 | 68.9 |
| B3 FocusViT-style, whole image, top-K 30% | 55.2 | | |
| B0 full image, 196 tokens | 76.1 | | |

The learned policy nearly reaches the ground-truth oracle and beats saliency by 16 points at 3 looks. **Caveat:** Stage B also fine-tunes M1/M3/M4 (joint lr 0.1×), so B1/B2 differ from B5 in more than the policy. Added `b1_random_on_b5`, `b2_saliency_on_b5` and `oracle_gt_on_b5` (policy is the only difference). The H1 comparisons now use those.

## 2026-11-23: full CUB matrix, seed 0

Test %, 25 tokens per look unless stated. Same-weights controls use the B5 checkpoint.

| Row | 1 look | 2 looks | 3 looks | Reading |
|---|---|---|---|---|
| random on B5 weights | 22.9 | 33.3 | 39.4 | |
| saliency on B5 weights | 38.4 | 45.6 | 50.7 | |
| **B5** | **58.8** | **64.2** | **66.5** | Learned aim: +15.8 over saliency with the same weights (H1, policy part ✓) |
| oracle gt on B5 weights | 59.6 | 65.7 | 68.6 | B5 is 2 points short of the oracle |
| A5 q = none | 58.4 | 64.5 | 66.4 | **The task signal adds nothing** (H1, task part ✗); the coarse head is only ~8% accurate, so q is nearly uninformative |
| A1 no offset PE | 58.7 | 64.3 | 65.8 | Negligible |
| A6 learned top-K, Stage A | 15.4 | 21.4 | 26.5 | Confirms the resize decision |
| B3 FocusViT-style, 1 look, top-K 30% | 55.2 | | | |
| B4 CAIT-style merge | 70.7 (full image, 1.41 GFLOPs) | | | |
| A4 coarse pass only | 8.0 | | | |
| B0 | 76.1 (196 tokens, 2.51 GFLOPs) | | | |

**H3 sweep (one checkpoint, τ* from val):** ρ 0.10 / 0.15 / 0.25 / 0.5 / 1.0 gives 54.4 / 66.5 / 74.7 / 79.8 / 80.4% at 0.37 / 0.53 / 0.90 / 1.41 / 2.22× B0 token-blocks. Above B0's cost the model beats B0; at equal cost (ρ=0.25) it is 1.4 points below. τ* is small (0.1–0.3), so entropy stopping rarely saves looks (2.5–2.9 mean looks).

**K2 fires.** Batch-1 latency, 3 looks, ρ=0.15 (1.38 GFLOPs vs 2.51):

| | RTX 3090 p50 | CPU p50 |
|---|---|---|
| B0 | 4.2 ms | 28.9 ms |
| B5 | 25.3 ms | 58.8 ms |

GPU breakdown for B5: ViT 13.9 ms (≈4.6 ms per 26-token look, about the same as a whole 197-token B0 pass, so launch-bound), policy 4.3, crop 2.9, embed 2.5, coarse 1.3. FLOP savings do not become time savings at batch 1. Candidate remedies: CUDA graphs / `torch.compile` per look, and fewer, larger looks.

**Canvas:** canvas B0 34.5% test; Stage A val 6.7% (its val uses random looks, which rarely hit a 32–56 px target); B5 val 47.8%. Looking pays off where the target is small.
