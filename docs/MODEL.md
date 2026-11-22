# SaccadeViT: what the method does and what it optimises

This file is the reference description of the method as implemented. Every symbol below maps to a file and a config key. When the code changes, update this file in the same commit. The reasoning behind the design lives in `../../DESIGN.md`. Choices still open are listed in `../../DECISIONS.md`.

## 1. Functionality

**Contract.** One classifier with three inference-time budget knobs:

```text
input   x ∈ R^{3×S×S}              S = 448 (CUB), 512 (canvas), 224 (CIFAR)
knobs   ρ ∈ (0,1]   token budget per look          model.keep_ratio  (default 0.15 → g=5, 25 tokens)
        K           maximum number of looks       model.k_max       (default 3)
        τ ∈ [0,1]   stop when normalised entropy < τ   eval.tau     (0 = never stop early)
output  p(y|x)      class distribution
        looks(x) ∈ {0..K}, the fixation trace (cell, scale, box) per look, and tokens used
```

All three knobs are set at inference. There is one checkpoint for every budget.

**The loop** (`models/saccade.py`), on one shared ViT-Ti/16 `f` with 12 blocks, N = 196 patches and width D = 192:

```text
M1 coarse    C = f_{1..2}(patch16(resize_224(x)))      C ∈ R^{N×D}, field of view = whole image
             g = LN(mean C);  p_0 = softmax(W_c g)     the coarse guess ("what it might be")
             q = W_q p_0  (stop-gradient on p_0)       the task signal (q = 0 in A5)
             e_0 = tanh(W_e g)
for t = 1..K:
  stop if  H(p_{t-1}) / log|Y| < τ                     (checked for t-1 = 0 too: zero looks is allowed)
  M2 aim     a_t = (cell, scale) ~ π_φ(· | C, g, e_{t-1}, q)
             cells ∈ 14×14 coarse grid (visited cells masked), scales ∈ {80,112,160,224} px
  M3 look    g = round(√(ρN));  crop_t = roi_align(x, box(a_t)) → (16g)²   (resize glimpse)
             crop_t ← shift each patch by (dx,dy) = 0.5·tanh(T(crop_t))   (patch translator)
             u_i = patch16(crop_t)_i + interp_g(pe)_i + MLP_off(dx_i,dy_i) + MLP_loc(abs_xy_i, side)
             z_t = LN(f_{1..12}([cls; u_1..u_{g²}]))_cls                  (all g² tokens, no selection)
  M4 accum   e_t = GRU(e_{t-1}, z_t);   p_t = softmax(W_h LN(e_t))
return p_t, t
```

| Piece | File | Origin |
|---|---|---|
| Coarse look with pretrained patch embedding at 448→224 | `models/coarse.py` | Choi 2022, DESIGN fix #3 |
| Policy over coarse cells × scales; inhibition of return | `models/policy.py` | **ours (addition 1)** |
| Resize glimpse (whole crop at g×g patches), translate, re-embed | `models/glimpse.py` | FocusViT translation + offset PE. **Resize replaces top-K** (EXPERIMENT_LOG findings 1–3); top-K is kept as A6/B3 |
| GRU accumulation + entropy stop | `models/aggregator.py` | **ours (addition 2)**, after AdaptiveNN |

**Cost model** (`engine/evaluate.py: at_tau`), in token-blocks relative to B0:

$$
\mathrm{cost}(x) = \frac{d_c N + \mathrm{looks}(x)\, d\, \lceil \rho N \rceil}{d N}, \qquad d_c = 2,\ d = 12
$$

Measured GFLOPs at 448² input (top-K version, 29 tokens per look): B0 = 2.51; 1.33 with 2 looks, 1.77 with 3. The resize glimpse at ρ = 0.15 feeds 25 tokens, slightly fewer. The claim itself is judged on batch-1 p50/p95 milliseconds (`engine/latency.py`), not on this proxy.

## 2. The problem being solved

The research question is a constrained one: spend as little computation as possible per image while staying within δ of the full-image model.

$$
\min_{\theta,\ \rho, K, \tau}\ \mathbb{E}_x\big[\mathrm{ms}(x)\big] \quad \text{s.t.}\quad \mathrm{Acc}(\theta,\rho,K,\tau) \ \ge\ \mathrm{Acc}_{B0} - \delta, \qquad \delta = 1\ \text{pt (slide 15)}
$$

Training does not optimise this directly. It optimises differentiable or sampled proxies (§3) for parameters θ. The budget knobs (ρ, K, τ) are then chosen without retraining on the validation split (§4). That split between training and inference is what H3 tests.

## 3. Training objectives (`engine/losses.py`)

Notation: $\ell_t = \log p_t(y)$, $\mathrm{CE}_\epsilon$ = cross-entropy with label smoothing $\epsilon = 0.1$. Optimiser: AdamW (wd 0.05) with cosine schedule, warm-up and gradient clip 1.0. bf16 autocast on CUDA.

### Stage 0: B0 backbone (`train.stage: full`)

$$
\min_{f}\ \mathbb{E}_{(x,y)}\ \mathrm{CE}_\epsilon\big(f(\mathrm{resize}_{224}(x)),\ y\big)
$$

Every parameter is trained. This checkpoint initialises every later stage. B4 uses the same loss with merging layers inserted.

### Stage A: learn to look, given where (`train.stage: a`)

Fixations come from a teacher, not from the policy. Per batch: $k \sim \mathcal U\{1..K\}$, $\rho \sim \mathcal U[0.1, 1]$, and per sample $a_t \sim$ mix(ground-truth box/parts 0.4, random 0.4, whole image 0.2).

$$
\mathcal L_A = \frac{1}{k}\sum_{t=1}^{k} \mathrm{CE}_\epsilon(p_t, y) \;+\; w_c\, \mathrm{CE}_\epsilon(p_0, y), \qquad w_c = 0.5
$$

- **Trained:** coarse head (W_c, LN), translator T, MLP_off, MLP_loc, GRU and head, and ViT blocks 3–12 at 0.1× learning rate (`train.tune_last_blocks: 10`).
- **Frozen:** the coarse blocks 1–2 (so the context map C stays fixed) and the policy π_φ.
- **Why randomise k and ρ:** so that one set of M3/M4 weights is valid for every budget swept later (fix #6).

**A6 only: how top-K stays differentiable.** For z-scored scores $\tilde s$, the training forward pass returns the averaged indicator over S = 16 noise draws (σ = 0.05):

$$
Y = \mathbb E_Z\big[\mathbf 1_{\mathrm{topk}}(\tilde s + \sigma Z)\big] \in \mathbb R^{k\times N}, \qquad \text{kept} = Y u
$$

with gradient $\partial Y/\partial \tilde s = \mathbb E_Z\big[\mathbf 1_{\mathrm{topk}}(\tilde s+\sigma Z)\, Z^\top\big]/\sigma$ (Berthet et al. 2020; reference code in `third_party/ptopk`). At inference this is replaced by a hard top-K and `gather`.

### Stage B: learn where to look (`train.stage: b`)

The policy is trained by REINFORCE with a self-reward: no fixation labels, only the label y the classifier already uses. Actions are sampled for K looks with $\rho \sim \mathcal U[0.1, 1]$.

$$
r_t = (\ell_t - \ell_{t-1}) - \lambda \frac{\lceil\rho N\rceil}{N}, \qquad
G_t = \sum_{s \ge t} r_s = \ell_K - \ell_{t-1} - \lambda \sum_{s\ge t} \frac{\lceil\rho N\rceil}{N}
$$

The reward telescopes, so each action is credited with the total improvement in the true class's log-probability from its look to the end. The whole-episode objective is therefore $\ell_K - \ell_0$ minus the token cost: "end more certain of the right class than the coarse pass was".

$$
\mathcal L_B = -\frac{1}{KB}\sum_{t,b} \hat A_{t,b}\, \log \pi_\phi(a_{t,b}) \;-\; \beta\, \bar H(\pi_\phi) \;+\; w_{ce}\, \frac1K \sum_t \mathrm{CE}_\epsilon(p_t, y)
$$

- **Advantage:** $\hat A_t$ = $G_t$ normalised across the batch at each step (mean as baseline, divided by the standard deviation).
- **Weights:** $\beta = 0.01$, $w_{ce} = 1$.
- **What moves:** π_φ at the full learning rate. M1 head, W_q, M3 and M4 at 0.1× (`train.joint_lr_scale`), and blocks 3–12 at 0.01×, so they adapt to the policy's fixation distribution.
- **Policy inputs are detached:** the policy gradient never reaches the backbone.
- **λ is currently inactive.** Every sample takes the same K looks at the same ρ, so the cost term is the same constant for the whole batch, and the per-step baseline cancels it. λ only matters once rollout length varies (a learned stop action). That is why it defaults to 0 and the stopping decision is made by τ instead.

## 4. Inference-time selection (`engine/sweep.py`)

No gradients. For each ρ, evaluate K looks once, then apply every τ after the fact:

$$
\mathrm{looks}_\tau(x) = \min\{t \ge 0 : H(p_t)/\log|Y| < \tau\} \wedge K, \qquad
\tau^* = \max\{\tau : \mathrm{Acc}_{val}(\tau) \ge \mathrm{Acc}_{val}(K) - \mathrm{tol}\}
$$

τ* is calibrated on val (10% of the official train split) and reported on test. `SaccadeViT.infer` applies the same rule with real early exit at batch 1, and that path is the one timed.

## 5. How each hypothesis reads these objectives

| | Compares | Same weights? | Make target |
|---|---|---|---|
| H1 | π_φ (B5) vs saliency (B2) vs random (B1) at equal (ρ, K) | B1/B2 use Stage A, B5 uses Stage B | `compare_h1_*` |
| H1 / A5 | q = W_q p_0 vs q = 0 | separate Stage-B runs from one Stage A | `compare_h1_task_signal` |
| H2 | τ* vs fixed K at matched accuracy | same B5 checkpoint | `compare_h2_stop` |
| H3 | the (ρ, K, τ) frontier in ms | one checkpoint | `sweep_a3_keep_sweep`, `latency_*` |

## 6. What is deliberately *not* optimised

- **When to stop** is a calibrated rule (τ), not learned. λ stays inert until a stop action exists.
- **The coarse blocks** f₁..₂ stay frozen after Stage 0, so the context map C is fixed.
- **The translator** (and the A6 scorer) are trained only through classification loss. There is no gaze or saliency supervision.
- **The policy** never sees the label. The label enters only through the reward.
