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

## 2026-11-23: CUB, 3 seeds, and hypothesis tests

Test %, mean ± std over seeds 0–2, 25 tokens per look. Full table: [CURRENT_RESULTS.md](CURRENT_RESULTS.md). Outputs: `output/<exp>/seed{0,1,2}/`.

| Row | 1 look | 2 looks | 3 looks |
|---|---|---|---|
| random on B5 weights | 22.9 ± 0.4 | 32.9 ± 0.7 | 39.5 ± 0.5 |
| saliency on B5 weights | 38.7 ± 0.2 | 46.5 ± 0.6 | 50.9 ± 0.2 |
| **B5** | **58.6 ± 0.9** | **64.2 ± 0.4** | **66.4 ± 0.3** |
| oracle gt on B5 weights | 60.5 ± 0.7 | 65.9 ± 0.2 | 68.8 ± 0.3 |
| A5 q = none | 59.0 ± 1.5 | 64.6 ± 0.6 | 66.4 ± 0.3 |
| A1 no offset PE | 57.7 ± 0.9 | 63.5 ± 0.6 | 65.5 ± 0.2 |
| A6 learned top-K | 13.7 ± 1.2 | 19.9 ± 1.1 | 25.0 ± 1.0 |
| B3 FocusViT-style (59 tokens) | 55.3 ± 1.1 | | |
| B4 merge | 71.0 ± 0.2 (full image) | | |
| A4 coarse only | 8.2 ± 0.2 | | |
| B0 | 76.3 ± 0.2 | | |

**Paired bootstrap** (3 seeds pooled, 2 looks; points, 95% CI). Output: `output/compare/<name>.json`.

| Test | Δ (b − a) | 95% CI | Reading |
|---|---|---|---|
| H1 `h1_task_vs_saliency`: B5 vs saliency, same weights | **+17.7** | [16.6, 18.8] | ✓ in every stratum; largest on large objects (+23.0) |
| H1 `h1_saliency_vs_random` | +13.7 | [12.5, 14.8] | Saliency beats random, most on small objects (+19.1) |
| H1/A5 `h1_task_signal`: q on vs off | −0.4 | [−0.9, 0.2] | ✗ no effect from the task signal |
| H2 `h2_stop`: τ=0.2 vs fixed K=3 | −0.04 | [−0.09, 0.01] | Equal accuracy with 2.3–2.9 mean looks instead of 3 (saves 5–25% of looks, seed-dependent) |

**H3 sweep** is stable across seeds: at ρ = 0.15 / 0.25 / 0.5, test 66.4 / 74.9 / 79.7% at 0.46–0.53 / 0.69–0.90 / 1.2–1.7× B0 token-blocks. τ* ranges over 0.1–0.3 per seed. B0 is reached only above ~1.2× B0 cost.

**Canvas (seed 0)**, test %, 3 looks: random 6.5, saliency 15.5, **B5 48.4**, B0 34.5. Here the learned policy beats the full image by 14 points with 75 tokens, because the 32–56 px target becomes tiny at 224².

## 2026-11-23: CIFAR-100 control and final latency

**CIFAR-100 (224², seed 0, test %):** B0 86.2. B5: 1 / 2 / 3 looks give 71.8 / 74.8 / 76.3; the coarse pass alone gives 36.7. With the object filling the frame there is nothing to find, so looking costs ~10 points: the control behaves as expected.

**Latency, batch 1, RTX 3090** (measured after all training had finished; the CPU numbers are therefore cleaner than the seed-0 ones):

| Model | GFLOPs | GPU p50 ms | CPU p50 ms |
|---|---|---|---|
| B0 | 2.51 | 4.19 | 20.6 |
| B4 merge | 1.41 | 4.79 | **12.0** |
| B3 FocusViT-style, 1 look | 1.23 | 7.64 | 25.8 |
| B5, K = 3 | 1.38 | – | 34.0 |
| B5, τ = 0.2 (2.64 looks) | 1.21 | 23.75 | – |

K2 still fires: ours is 5.7× slower on GPU and 1.6× slower on CPU, while B4 (the CAIT-style merge) is 1.7× *faster* on CPU. On GPU, the ViT part alone takes 11.2 ms for 2.6 small looks, and the policy takes 3.2 ms.

## 2026-11-24: phase 2, accuracy vs time on newpc (RTX 5090 D)

Plan and claims: [TRADEOFF.md](TRADEOFF.md). All GPU numbers use bf16 autocast and CUDA graphs for every model (`eval.graph=true`). Outputs: `output/<exp>/seed0/latency_cuda_graph*.json`. The generated Pareto table is in [CURRENT_RESULTS.md](CURRENT_RESULTS.md). newpc CPU timings were discarded: the VM's CPUs gave about 1 s per image for B0, so the phase-1 CPU numbers from hitcvsz4 stand.

**New runs (CUB, 3 seeds)**

| Run | Test % | Note |
|---|---|---|
| B0 native 448 (`b0_448`, 784 tokens) | **84.08 ± 0.22** | The resolution ceiling, +7.8 over B0-224 |
| B5-par (`b5_par`, parallel looks), ρ=0.15 | 59.3 / 64.7 / 66.4 ± 0.2 at K=1/2/3 | Same accuracy as sequential B5 (58.6 / 64.2 / 66.4) |

**T1: graphs remove the launch overhead, but batch-1 time is still set by kernel count** (ms, batch 1)

| Model | Eager | CUDA graph |
|---|---|---|
| B0 224 | – | 0.72 |
| B0 native 448 | – | 0.85 |
| B4 merge | – | 0.69 |
| B5, 3 looks, ρ=0.15 | – | 2.22 (2.29 before caching) |
| **B5-par**, 3 looks, ρ=0.15 | – | **1.30** |

At batch 1 a 5090 is nearly idle even for 784 tokens: B0 at 448 costs only +0.13 ms over B0-224. Time follows the number of sequential kernels, so a multi-pass method cannot win in this regime. **The batch-1 GPU claim is dropped for good** (TRADEOFF.md).

**T2/T3: throughput (batch 64) is where token budgets pay off.** Pareto front over (test acc, mean of 3 seeds; images/s, seed 0):

| Operating point | Tokens | Test % | img/s | On the front? |
|---|---|---|---|---|
| B0 native 448 | 784 | 84.08 | 5,762 | ✓ |
| **B5-par ρ=1.0 K=2** | 2×196 | 80.19 | 8,233 | ✓ |
| **B5-par ρ=0.5 K=3** | 3×100 | 79.99 | 10,155 | ✓ |
| **B5-par ρ=0.5 K=2** | 2×100 | **79.13** | **13,682** | ✓ |
| B0 224 | 196 | 76.30 | 19,394 | ✓ |
| B4 merge | 98→49 | 70.96 | 23,772 | ✓ |
| **B5-par / B5 ρ=0.15 K=1** | 1×25 | 58.6–59.3 | 25,500 | ✓ |

Reading:
- Between B0-224 and B0-native, ours is the only option. For example, B5-par at ρ=0.5, K=2 gains +2.8 points over B0-224 at 2.4× the throughput of B0-native. One checkpoint covers the whole front by setting (ρ, K) at test time.
- Parallel looks are better than sequential at every (ρ, K): +0–2 points of throughput-equivalent accuracy, and 1.3–1.9× faster at batch 1.
- Ours does **not** beat B0-224 at B0-224's own throughput (76.1% at 16.7k vs 76.3% at 19.4k), nor B4 below it. The win is the interpolation towards native-resolution accuracy.
- Canvas (small objects) and the 768² canvas (T4, T5) are not run yet. They need CIFAR on newpc (~40 min download) and about 3 h of GPU time.

**T6: small objects on real data (CUB, test split into object-area terciles, paired bootstrap, seed 0).** Output: `output/compare/t6_*.json`.

| Comparison (Δ = ours − baseline) | All | Small third | Medium | Large |
|---|---|---|---|---|
| B5 ρ=0.15 K=3 vs B0-224 (3 seeds) | −9.9 [−10.6, −9.1] | −10.1 | −9.3 | −10.2 |
| **B5-par ρ=0.5 K=2 vs B0-224** | **+2.9** [2.0, 3.9] | **+8.3** [6.5, 10.1] | +2.5 | −2.1 |
| B5-par ρ=0.5 K=2 vs B0-native 448 | −5.2 | −4.6 | −4.1 | −7.0 |
| **B5-par ρ=0.5 K=2 vs B0-224, 3 seeds pooled** (2026-11-25) | **+2.8** [2.2, 3.4] | **+7.9** [6.7, 9.2] | +2.7 [1.7, 3.7] | −2.3 [−3.3, −1.3] |

At the trade-off operating point, the whole gain over B0-224 comes from small and medium birds; large birds lose a little. The default budget (ρ=0.15) is too small to show any size effect.

**2026-11-24 afternoon, canvas speed-up for newpc.**
- Canvas images now travel as uint8 and are normalised on the GPU (`Backbone.prep`).
- The bicubic background is computed as two exact matrix products (max error 6e-7).
- Result: 3.5× faster generation, and 4× less data per sample. Test canvases are unchanged.

## 2026-11-24 evening: T4 canvas 512² on PC48 (user2), seed 0

Platform: PC48, an RTX 5090 D like PC23, after PC23 dropped offline. Outputs: `output/canvas-*/seed0/`. Accuracy is on the test split; throughput is batch 64 with CUDA graphs.

| Model / operating point | Tokens | Test % | img/s (b64) | GPU ms (b1) |
|---|---|---|---|---|
| B0 native 512 (`canvas/b0_native`) | 1024 | 60.37 | 3,980 | 0.82 |
| B5 ρ=1.0 K=3 | 3×196 | 59.50 | 5,585 | 2.48 |
| **B5-par ρ=0.5 K=3** | 3×100 | **58.72** | **9,850** | 1.21 |
| B5-par ρ=0.5 K=2 | 2×100 | 55.87 | 12,617 | 1.12 |
| B5-par ρ=0.15 K=3 | 3×25 | 48.32 | 14,950 | 1.18 |
| **B5 ρ=0.15 K=1** | 1×25 | **40.10** | **20,748** | 0.94 |
| B0 224 | 196 | 34.45 | 17,166 | 0.66 |
| B4 merge | 98→49 | 33.54 | 19,421 | 0.64 |
| B2 saliency looks, 3 looks | 3×25 | 15.42 | | |
| B1 random looks, 3 looks | 3×25 | 6.64 | | |

**The small-object goal is met on throughput.**
- Our cheapest point (1 look, 25 tokens) beats both B0-224 (+5.7 pt) and B4 (+6.6 pt) **at higher throughput** than either.
- At the top end, B5-par comes within 1.6 points of native-resolution B0 at 2.5× its throughput. Sequential B5 at ρ=1.0, K=3 comes within 0.9 points at 1.4×.
- At batch 1 ours remains slower than B0 (0.94 ms vs 0.66 ms).
- Seeds 1–2 and the 768² canvas are running.

## 2026-11-24 night: T4 canvas 512², 3 seeds (PC48)

Test %, mean ± std over seeds 0–2. Throughput is from seed 0 (batch 64, CUDA graphs).

| Model / operating point | Test % | img/s (b64) |
|---|---|---|
| B0 native 512 | 59.67 ± 0.58 | 3,980 |
| B5 ρ=1.0 K=3 | 59.59 ± 0.32 | 5,585 |
| **B5-par ρ=0.5 K=3** | **59.18 ± 0.51** | **9,850** |
| B5-par ρ=0.5 K=2 | 56.17 ± 0.32 | 12,617 |
| B5-par ρ=0.25 K=2 | 52.50 ± 0.34 | 14,312 |
| B5-par ρ=0.15 K=3 | 47.98 ± 0.31 | 14,950 |
| **B5 ρ=0.15 K=1** | **39.62 ± 0.53** | **20,748** |
| B0 224 | 34.56 ± 0.35 | 17,166 |
| B4 merge | 33.66 ± 0.13 | 19,421 |
| B2 saliency, 3 looks | 15.25 ± 0.13 | |
| B1 random, 3 looks | 6.41 ± 0.22 | |

Seeds 1 and 2 confirm seed 0, with differences well beyond the seed spread:
- B5 ρ=0.15 K=1 beats B0-224 by +5.1 pt and B4 by +6.0 pt, at higher throughput than either.
- B5-par ρ=0.5 K=3 is within 0.5 pt of native-resolution B0 (inside one std) at 2.5× its throughput.
- With 3 looks, learned aiming beats saliency by +32.7 pt and random by +41.6 pt.

## 2026-11-25: T5 canvas 768², seed 0 (new RTX 4090 machine)

PC48 went offline, so this ran on a rented RTX 4090. Its speeds are comparable only with each other, not with the 512² table from the 5090. The targets are still 32–56 px, so each object is a smaller fraction of the image. Outputs: `output/canvas768-*/seed0/`.

| Model / operating point | Tokens | Test % | img/s (b64) | GPU ms (b1) |
|---|---|---|---|---|
| B0 native 768 | 2304 | 57.55 | 979 | 1.34 |
| B5 ρ=1.0 K=3 | 3×196 | 52.54 | 4,153 | 2.32 |
| **B5 ρ=0.5 K=3** | 3×100 | **51.36** | **6,103** | 2.10 |
| B5 ρ=0.5 K=2 | 2×100 | 47.57 | 7,581 | 1.52 |
| B5 ρ=1.0 K=1 | 1×196 | 40.45 | 7,976 | **1.05** |
| B5 ρ=0.5 K=1 | 1×100 | 39.63 | 10,019 | 0.90 |
| **B5 ρ=0.15 K=1** | 1×25 | **30.56** | **11,671** | 0.92 |
| B0 224 | 196 | 24.90 | 9,863 | 0.62 |

**What changes from 512² to 768²:**
- The win at the cheap end is the same. One 25-token look beats B0-224 by +5.7 pt at 1.18× its throughput.
- B0 at native resolution costs much more now: 2,304 tokens, 979 img/s. Ours at ρ=0.5 K=3 runs at **6.2× its throughput** and gives up 6.2 pt. At 512² the cost was 0.5 pt for 2.5×. So the gap to native B0 widens with resolution, at a much larger speed-up.
- At **batch 1**, ours is now on the time front too. ρ=1.0 K=1 (1.05 ms, 40.5%) and ρ=0.5 K=1 (0.90 ms, 39.6%) are faster than B0-native's 1.34 ms. They are also 15 pt more accurate than B0-224. This is F5: our cost does not grow with the input size, while B0's does.
- Seed 0 only. B5-par and B4 were not trained at 768².

## 2026-11-25 evening: T5 canvas 768², 3 seeds (RTX 4090)

Test %, mean ± std over seeds 0–2. Throughput and ms are from seed 0 (speed does not depend on the seed).

| Model / operating point | Test % | img/s (b64) | GPU ms (b1) |
|---|---|---|---|
| B0 native 768 | 57.44 ± 0.14 | 979 | 1.34 |
| B5 ρ=1.0 K=3 | 52.50 ± 0.20 | 4,153 | 2.32 |
| **B5 ρ=0.5 K=3** | **51.77 ± 0.40** | **6,103** | 2.10 |
| B5 ρ=0.5 K=2 | 48.10 ± 0.51 | 7,581 | 1.52 |
| B5 ρ=1.0 K=1 | 41.07 ± 0.54 | 7,976 | 1.05 |
| B5 ρ=0.5 K=1 | 40.42 ± 0.69 | 10,019 | 0.90 |
| **B5 ρ=0.15 K=1** | **31.61 ± 1.04** | **11,671** | 0.92 |
| B0 224 | 24.91 ± 0.06 | 9,863 | 0.62 |

Seeds 1 and 2 confirm seed 0:
- One 25-token look beats B0-224 by +6.7 pt, at 1.18× its throughput.
- Ours at ρ=0.5 K=3 runs at 6.2× native B0's throughput, for −5.7 pt.
- At batch 1, one look (0.90–1.05 ms) is faster than B0-native (1.34 ms) and 15–16 pt more accurate than B0-224.
