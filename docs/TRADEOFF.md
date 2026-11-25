# Phase 2: the accuracy–time trade-off

Agreed scope (2026-11-24). Success means two things: **beat the baselines on small objects**, and give **a solid accuracy-vs-milliseconds trade-off**. Strict novelty is not required. The results of these experiments are logged in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Claims kept and dropped

| Claim | Status | Why |
|---|---|---|
| Learned aim beats saliency and random at equal budget | **kept** (proven, +17.7 pt) | |
| Wins where objects are small | **kept**, main claim | canvas +14 pt over B0 |
| One checkpoint, budget set at test time | **kept** | |
| Accuracy vs ms Pareto against B0 / B0-high-res / B4 | **new main claim**, measured as **throughput** (batch 64) | At batch 1 the 5090 is idle even at 784 tokens (B0-448 0.85 ms vs B0-224 0.72 ms), so time follows kernel count and no multi-pass method can win there. On the throughput front, B5-par holds every point between B0-224 and B0-native (EXPERIMENT_LOG, 2026-11-24) |
| Looks are *task*-conditioned (`q`) | **dropped** | −0.4 pt, CI includes 0. `q` stays in the code, but no claim is made |
| Entropy stopping saves compute (H2) | **dropped as a claim** | It saves only 5–25% of looks. τ stays as a knob |
| ≥1.8× faster than B0-224 at batch 1, eager PyTorch | **dropped** | ViT-Ti at ≤197 tokens is launch-bound, so no token reduction can win there. Replaced by the Pareto claim with CUDA graphs for every model |

## Why the method loses time, and what each fix targets

At batch 1 on GPU, one ViT-Ti pass costs about 4 ms whether it carries 26 tokens or 197. The cost is kernel launches, not arithmetic. B5 makes 1 coarse pass and 3 sequential looks, plus a policy built from many small ops, so it pays that fixed cost about 4 times.

| Fix | What it changes | Expected effect |
|---|---|---|
| **F1. CUDA graphs** for every model (B0, B4, ours) | Removes launch overhead; time then follows FLOPs | Ours becomes compute-bound. A fair comparison, since baselines get graphs too |
| **F2. Greedy fast path** in the policy (no `Categorical`, no sync) | Policy 3.2 ms → ~0 | Needed for F1 (no host syncs inside a graph) |
| **F3. Parallel looks** (`model.parallel: true`) | Picks all K fixations from the coarse map at once, then runs the K crops as **one batched ViT pass**. The GRU still fuses them in order | 2 passes instead of 1 + K. Trades sequential re-aiming for speed |
| **F4. Fewer, larger looks** | Choose (ρ, K) on the Pareto front, e.g. 1 look at ρ = 0.5 | Measured, not assumed |
| **F5. Resolution independence** | Our cost does not grow with input size. B0 at native resolution grows as tokens² | The trade-off wins at high resolution and on small objects |

## Experiments

All latency figures: batch 1 p50 on the RTX 5090 D (CUB, canvas 512²) or a rented RTX 4090 (canvas 768²), CUDA graphs on (`eval.graph=true`), bf16. Eager numbers are reported too. Throughput at batch 64 (images/s) is a secondary column.

| ID | Question | Runs | Make target |
|---|---|---|---|
| **T1** | Do graphs close the gap? | Latency for B0, B4, B3, B5 with graphs on and off | `latency_graph_all` |
| **T2** | Pareto on CUB | B5 at (ρ, K) ∈ {0.15, 0.25, 0.5, 1.0} × {1, 2, 3}: accuracy from the sweep, ms from the latency grid, against B0-224, **B0-448**, B4 | `train_b0_448`, `latency_grid_b5_ours` |
| **T3** | Parallel looks | Stage B with `parallel: true` (`b5_par`), 3 seeds; same grid | `train_b5_par`, `latency_grid_b5_par` |
| **T4** | Small objects, full statistics | Canvas: B0-224, **B0-512 native** (1024 tokens), B4, B1, B2, B5, B5-par, 3 seeds | `canvas_3_seeds` |
| **T5** | Object size vs resolution | Canvas at 768² with fixed 32–56 px targets (smaller fraction of the image): B0-224, B0-native, B5 | `canvas768` |
| **T6** | Small-object stratum on real data | CUB test split into object-area terciles: B5 vs B0 vs B4, paired bootstrap, no new training | `compare_small_*` |

**Decision rule.** For each dataset, report the Pareto set over (accuracy, GPU ms). The trade-off is a success if some B5 / B5-par operating point is (a) more accurate than every baseline at no more ms, or (b) faster than every baseline at no lower accuracy.

## Verdict (2026-11-25)

Both success conditions hold. Numbers are 3 seeds unless marked; details in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

| Goal | Evidence | Met |
|---|---|---|
| Beat the baselines on small objects | Canvas 512²: one 25-token look gives 39.6 ± 0.5% vs B0-224 34.6 ± 0.4% and B4 33.7 ± 0.1%, at higher throughput than both. Canvas 768² (seed 0): +5.7 pt over B0-224, again faster. CUB T6: +7.9 pt [6.7, 9.2] over B0-224 on the smallest third of birds | ✓ |
| Solid accuracy-vs-time trade-off (decision rule, throughput) | (a) CUB: B5-par holds every Pareto point between B0-224 and B0-448. (b) Canvas 512²: B5-par ρ=0.5 K=3 is within 0.5 pt of native B0 at 2.5× its throughput, and B5 ρ=0.15 K=1 is faster *and* more accurate than B0-224 and B4. (c) Canvas 768²: 6.2× native B0's throughput at −6.2 pt | ✓ |
| Batch-1 milliseconds | Not a claim (dropped). It still holds at 768²: 1 look beats B0-native on ms and B0-224 on accuracy | partial |

Limits to state in the write-up:
- Ours never beats B0-224 on CUB at B0-224's own throughput. The CUB win is the interpolation towards native-resolution accuracy.
- Large CUB birds lose 2.3 pt.
- Canvas 768² is a single seed.
- The canvas 512² checkpoints for B5-par seeds 1–2 were lost with PC48. Their results are kept.
