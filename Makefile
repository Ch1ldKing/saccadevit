.DEFAULT_GOAL := help
UV ?= uv
SV := $(UV) run sv
SEED ?= 0
ARGS ?=
E := configs/exp
export PYTHONDONTWRITEBYTECODE := 1

# Usage: make <target> [SEED=1] [ARGS="train.epochs=5 ..."]; `make -n <target>` previews.
# Run order on CUB: train_b0_full -> train_stage_a -> evaluate_* (B1-B3, oracle) -> train_b5_ours -> ...
# `make preliminary` = the seed-0 subset for first results; `make cub` = the whole CUB matrix for SEED.

.PHONY: help setup input_fetch input_verify third_party_fetch weights_fetch validate_configurations test report
help:
	@echo "make <target> [SEED=k] [ARGS='key=value ...'];  make -n <target> previews"
	@grep -E '^[a-z0-9_]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-34s %s\n", $$1, $$2}'
	@printf "  %-34s %s\n" "evaluate_<exp>" "test-set eval, <exp> in: $(CUB_EVALS)"
	@printf "  %-34s %s\n" "latency_{gpu,cpu}_<exp>" "batch-1 p50/p95 + per-module ms, <exp> in: $(LATENCY)"

setup: ## uv sync + download datasets and weights into input/ + snapshot third_party/
	$(UV) sync
	$(SV) assets fetch
	$(SV) assets verify

input_fetch: ## datasets and weights only (third_party/ is committed)
	$(SV) assets fetch --only datasets weights

weights_fetch: ## pretrained ViT weights only
	$(SV) assets fetch --only weights

third_party_fetch: ## re-snapshot third_party/ at the commits pinned in configs/assets.yaml
	$(SV) assets fetch --only third_party

input_verify: ## file counts + SHA-256 of everything fetched
	$(SV) assets verify

validate_configurations: ## load all experiment configs, build every model, check checkpoint links
	$(SV) validate

report: ## collect output/ into docs/CURRENT_RESULTS.md (configs, results, paths)
	$(SV) report

test: ## lean test suite on fake CUB (CPU)
	$(UV) run pytest -q

# ---------------------------------------------------------------- CUB-200-2011 (main)
.PHONY: train_a6_topk_stage_a train_b0_full train_stage_a train_b4_merge train_b5_ours train_a5_no_task train_a1_no_offset_stage_a train_a1_no_offset
train_b0_full: ## B0 / stage 0: full image, all 196 tokens (gate M1)
	$(SV) train $(E)/b0_full.yaml seed=$(SEED) $(ARGS)
train_stage_a: ## stage A: M3 + M4 on teacher fixations (needs b0_full)
	$(SV) train $(E)/stage_a.yaml seed=$(SEED) $(ARGS)
train_b4_merge: ## B4: CAIT-style merging (needs b0_full)
	$(SV) train $(E)/b4_merge.yaml seed=$(SEED) $(ARGS)
train_b5_ours: ## B5 / stage B: REINFORCE task-conditioned policy (needs stage_a)
	$(SV) train $(E)/b5_ours.yaml seed=$(SEED) $(ARGS)
train_a5_no_task: ## A5: stage B with q switched off (needs stage_a)
	$(SV) train $(E)/a5_no_task.yaml seed=$(SEED) $(ARGS)
train_a1_no_offset_stage_a: ## A1 part 1: stage A without offset PE (needs b0_full)
	$(SV) train $(E)/a1_no_offset_stage_a.yaml seed=$(SEED) $(ARGS)
train_a1_no_offset: ## A1 part 2: stage B on the no-offset stage A
	$(SV) train $(E)/a1_no_offset.yaml seed=$(SEED) $(ARGS)
train_a6_topk_stage_a: ## A6: stage A with learned top-K instead of resize glimpses (also B3's checkpoint)
	$(SV) train $(E)/a6_topk_stage_a.yaml seed=$(SEED) $(ARGS)

CUB_EVALS := b0_full b4_merge a6_topk_stage_a b1_random b2_saliency b3_focusvit oracle_gt b5_ours b1_random_on_b5 b2_saliency_on_b5 oracle_gt_on_b5 a1_no_offset a2_fixed_k a4_coarse_only a5_no_task
.PHONY: $(addprefix evaluate_,$(CUB_EVALS)) sweep_a3_keep_sweep
$(addprefix evaluate_,$(CUB_EVALS)): evaluate_%: ## test-set eval -> output/<exp>/seed<k>/summary_test.json
	$(SV) eval $(E)/$*.yaml seed=$(SEED) $(ARGS)
evaluate_cub_all: $(addprefix evaluate_,$(CUB_EVALS)) ## every CUB evaluation for SEED
sweep_a3_keep_sweep: ## A3 / H2 / H3: keep ratio x tau sweep on one B5 checkpoint, tau* from val
	$(SV) sweep $(E)/a3_keep_sweep.yaml seed=$(SEED) $(ARGS)

# ---------------------------------------------------------------- latency (K2), batch 1
LATENCY := b0_full b3_focusvit b4_merge b5_ours
.PHONY: $(addprefix latency_gpu_,$(LATENCY)) $(addprefix latency_cpu_,$(LATENCY)) latency_gpu_b5_ours_stop latency_all
$(addprefix latency_gpu_,$(LATENCY)): latency_gpu_%: ## p50/p95, per-module ms, GFLOPs on CUDA
	$(SV) latency $(E)/$*.yaml seed=$(SEED) eval.device=cuda $(ARGS)
$(addprefix latency_cpu_,$(LATENCY)): latency_cpu_%: ## the same on CPU
	$(SV) latency $(E)/$*.yaml seed=$(SEED) eval.device=cpu eval.iters=200 $(ARGS)
latency_gpu_b5_ours_stop: ## B5 with real early stopping (set TAU from sweep.csv tau*)
	$(SV) latency $(E)/b5_ours.yaml seed=$(SEED) eval.device=cuda eval.tau=$(or $(TAU),0.4) $(ARGS)
latency_all: $(addprefix latency_gpu_,$(LATENCY)) $(addprefix latency_cpu_,$(LATENCY)) latency_gpu_b5_ours_stop

# ---------------------------------------------------------------- hypothesis tests (3 seeds pooled)
.PHONY: compare_h1_task_vs_saliency compare_h1_saliency_vs_random compare_h1_task_signal compare_h2_stop
compare_h1_task_vs_saliency: ## H1: B5 vs saliency on the same B5 weights, same budget
	$(SV) compare 'output/b2_saliency_on_b5/seed*/eval_test.npz' 'output/b5_ours/seed*/eval_test.npz' --rule k=2 --name h1_task_vs_saliency
compare_h1_saliency_vs_random: ## H1: saliency vs random on the same B5 weights
	$(SV) compare 'output/b1_random_on_b5/seed*/eval_test.npz' 'output/b2_saliency_on_b5/seed*/eval_test.npz' --rule k=2 --name h1_saliency_vs_random
compare_h1_task_signal: ## H1 / A5: same model, q off vs on
	$(SV) compare 'output/a5_no_task/seed*/eval_test.npz' 'output/b5_ours/seed*/eval_test.npz' --rule k=2 --name h1_task_signal
compare_h2_stop: ## H2: same B5 checkpoint, fixed K_max vs entropy stop (TAU = tau* from sweep.csv)
	$(SV) compare 'output/a2_fixed_k/seed*/eval_test.npz' 'output/b5_ours/seed*/eval_test.npz' --rule-a last --rule-b tau=$(or $(TAU),0.4) --name h2_stop

# ---------------------------------------------------------------- second task (canvas) and control (CIFAR)
.PHONY: train_canvas_b0_full train_canvas_stage_a train_canvas_b5_ours evaluate_canvas_all train_cifar_b0_full train_cifar_stage_a train_cifar_b5_ours evaluate_cifar_all
train_canvas_b0_full: ## canvas B0 (run first as a pipeline check: "where" matters by construction)
	$(SV) train $(E)/canvas/b0_full.yaml seed=$(SEED) $(ARGS)
train_canvas_stage_a: ## canvas stage A
	$(SV) train $(E)/canvas/stage_a.yaml seed=$(SEED) $(ARGS)
train_canvas_b5_ours: ## canvas B5
	$(SV) train $(E)/canvas/b5_ours.yaml seed=$(SEED) $(ARGS)
evaluate_canvas_all: ## canvas B0, B1, B2, B5
	for x in b0_full b1_random b2_saliency b5_ours; do $(SV) eval $(E)/canvas/$$x.yaml seed=$(SEED) $(ARGS) || exit 1; done
train_cifar_b0_full: ## CIFAR-100 control B0
	$(SV) train $(E)/cifar/b0_full.yaml seed=$(SEED) $(ARGS)
train_cifar_stage_a: ## CIFAR-100 control stage A
	$(SV) train $(E)/cifar/stage_a.yaml seed=$(SEED) $(ARGS)
train_cifar_b5_ours: ## CIFAR-100 control B5
	$(SV) train $(E)/cifar/b5_ours.yaml seed=$(SEED) $(ARGS)
evaluate_cifar_all: ## CIFAR-100 control B0, B5
	for x in b0_full b5_ours; do $(SV) eval $(E)/cifar/$$x.yaml seed=$(SEED) $(ARGS) || exit 1; done

# ---------------------------------------------------------------- pipelines (dependency order)
.PHONY: canvas preliminary cub cub_3_seeds
canvas: train_canvas_b0_full train_canvas_stage_a train_canvas_b5_ours evaluate_canvas_all ## canvas end to end for SEED

preliminary: ## seed-0 first results: M1, M2, B1-B3/oracle, B5, latency
	$(MAKE) train_b0_full train_stage_a train_a6_topk_stage_a SEED=0
	$(MAKE) evaluate_b0_full evaluate_b1_random evaluate_b2_saliency evaluate_b3_focusvit evaluate_oracle_gt SEED=0
	$(MAKE) train_b5_ours evaluate_b5_ours sweep_a3_keep_sweep SEED=0
	$(MAKE) latency_gpu_b0_full latency_gpu_b5_ours latency_cpu_b0_full latency_cpu_b5_ours SEED=0

cub: ## the whole CUB matrix for SEED
	$(MAKE) train_b0_full train_stage_a train_a6_topk_stage_a train_b5_ours train_a5_no_task train_a1_no_offset_stage_a train_a1_no_offset train_b4_merge SEED=$(SEED)
	$(MAKE) evaluate_cub_all sweep_a3_keep_sweep SEED=$(SEED)

cub_3_seeds: ## CUB matrix for seeds 0 1 2, then latency and H1/H2 comparisons
	for s in 0 1 2; do $(MAKE) cub SEED=$$s || exit 1; done
	$(MAKE) latency_all
	$(MAKE) compare_h1_task_vs_saliency compare_h1_saliency_vs_random compare_h1_task_signal compare_h2_stop

# ---------------------------------------------------------------- phase 2: accuracy-time trade-off (docs/TRADEOFF.md)
GRID_RHO ?= 0.1 0.15 0.25 0.5 1.0
GRID_K ?= 1 2 3
LAT_GRAPH := $(SV) latency
.PHONY: train_b0_448 train_b5_par sweep_a3_keep_sweep_par evaluate_b0_448 evaluate_b5_par latency_graph_all latency_grid_b5_ours latency_grid_b5_par throughput_all tradeoff_cub
train_b0_448: ## T2: B0 on the native 448^2 image (784 tokens)
	$(SV) train $(E)/b0_448.yaml seed=$(SEED) $(ARGS)
train_b5_par: ## T3: stage B with parallel looks (needs stage_a)
	$(SV) train $(E)/b5_par.yaml seed=$(SEED) $(ARGS)
evaluate_b0_448 evaluate_b5_par: evaluate_%:
	$(SV) eval $(E)/$*.yaml seed=$(SEED) $(ARGS)
sweep_a3_keep_sweep_par: ## T3: rho x K sweep of B5-par
	$(SV) sweep $(E)/a3_keep_sweep_par.yaml seed=$(SEED) $(ARGS)
latency_graph_all: ## T1: batch-1 GPU latency with CUDA graphs, every CUB model (+ eager B0/B5 for reference)
	for x in b0_full b0_448 b4_merge b3_focusvit b5_ours b5_par; do $(LAT_GRAPH) $(E)/$$x.yaml seed=$(SEED) eval.device=cuda eval.graph=true eval.half=true || exit 1; done
	for x in b0_full b0_448 b4_merge b5_ours b5_par; do $(LAT_GRAPH) $(E)/$$x.yaml seed=$(SEED) eval.device=cuda eval.half=true || exit 1; done
latency_grid_b5_ours latency_grid_b5_par: latency_grid_%: ## T2/T3: graph latency over rho x K
	for r in $(GRID_RHO); do for k in $(GRID_K); do $(LAT_GRAPH) $(E)/$*.yaml seed=$(SEED) eval.device=cuda eval.graph=true eval.half=true eval.keep_ratio=$$r model.k_max=$$k eval.iters=300 || exit 1; done; done
throughput_all: ## secondary: images/s at batch 64 with graphs
	for x in b0_full b0_448 b4_merge b5_ours b5_par; do $(LAT_GRAPH) $(E)/$$x.yaml seed=$(SEED) eval.device=cuda eval.graph=true eval.half=true eval.latency_batch=64 eval.latency_images=8 eval.warmup=10 eval.iters=50 || exit 1; done
tradeoff_cub: ## T1-T3 for SEED: B0-448, B5-par, sweeps, then latency (seed 0 only)
	$(MAKE) train_b0_448 train_b5_par evaluate_b0_448 evaluate_b5_par sweep_a3_keep_sweep_par SEED=$(SEED)

CANVAS_TRAIN := b0_full b0_native stage_a b4_merge b5_ours b5_par
CANVAS_EVAL := b0_full b0_native b4_merge b1_random b2_saliency b5_ours b5_par
.PHONY: canvas_full canvas_3_seeds latency_canvas canvas768
canvas_full: ## T4: whole canvas matrix for SEED (skips finished runs)
	for x in $(CANVAS_TRAIN); do $(SV) train $(E)/canvas/$$x.yaml seed=$(SEED) $(ARGS) || exit 1; done
	for x in $(CANVAS_EVAL); do $(SV) eval $(E)/canvas/$$x.yaml seed=$(SEED) $(ARGS) || exit 1; done
	$(SV) sweep $(E)/canvas/keep_sweep.yaml seed=$(SEED) $(ARGS)
	$(SV) sweep $(E)/canvas/keep_sweep_par.yaml seed=$(SEED) $(ARGS)
canvas_3_seeds: ## T4 for seeds 0 1 2
	for s in 0 1 2; do $(MAKE) canvas_full SEED=$$s || exit 1; done
latency_canvas: ## T4: graph latency on the canvas (B0, B0-native, B4, B5 / B5-par grid)
	for x in b0_full b0_native b4_merge; do $(LAT_GRAPH) $(E)/canvas/$$x.yaml seed=$(SEED) eval.device=cuda eval.graph=true eval.half=true || exit 1; done
	for x in b5_ours b5_par; do for r in $(GRID_RHO); do for k in $(GRID_K); do $(LAT_GRAPH) $(E)/canvas/$$x.yaml seed=$(SEED) eval.device=cuda eval.graph=true eval.half=true eval.keep_ratio=$$r model.k_max=$$k eval.iters=300 || exit 1; done; done; done
canvas768: ## T5: 768^2 canvas, fixed 32-56 px targets: B0, B0-native, stage A, B5, sweep
	for x in b0_full b0_native stage_a b5_ours; do $(SV) train $(E)/canvas768/$$x.yaml seed=$(SEED) $(ARGS) || exit 1; done
	for x in b0_full b0_native b5_ours; do $(SV) eval $(E)/canvas768/$$x.yaml seed=$(SEED) $(ARGS) || exit 1; done
	$(SV) sweep $(E)/canvas768/keep_sweep.yaml seed=$(SEED) $(ARGS)
