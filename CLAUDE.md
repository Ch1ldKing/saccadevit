# CLAUDE.md: SaccadeViT

The codebase for Group 3's NeuroAI project 3. Docs live outside this repo:
- `../DESIGN.md`: the method, the 8 fixes to the slides, and the build order. **Source of truth for *why*.** If code and DESIGN.md disagree, update DESIGN.md in the same change.
- `../../Saccadic_ViT_Presentation_EN 1.pptx`: the slides (research questions, baselines, targets, kill criteria).
- `../../digests/总报告.md`, `../../dives/`: literature notes, including fallback routes A1/A3/D3.
- `README.md`: commands and the experiment table. Keep it in sync when configs are added.

## Rules
- `uv` only (`uv sync`, `uv run sv ...`, `uv run pytest -q`). Run from the repo root, since paths in configs are relative.
- One YAML per experiment in `configs/exp/` with `_base_` inheritance. Never fork code for an experiment. Add a config key instead.
- Baselines and ablations share one action space and one Stage-A checkpoint, which is what keeps comparisons at equal budget. Keep it that way.
- Token reduction must use `gather` at inference, never masks. Otherwise the latency claim is invalid.
- Minimal defensive code. Lean tests: `tests/test_modules.py` (unit), `tests/test_pipeline.py` (every stage end-to-end on fake CUB, CPU, ~10 s).
- Platform: one L40 48 GB. Latency is reported at batch 1, p50/p95, on both GPU and CPU.

## Decisions made in code (beyond DESIGN.md)
- `q` defaults to `coarse_logits` (embedding of the coarse class belief). Attribute queries are not built yet.
- A4 = `k_max: 0` (coarse pass only). B3 = one whole-image look (scale index `len(scales)`). Stage A's teacher mixes gt / random / full view, so B3 is in-distribution.
- Patch translator = light stride-16 conv over the crop, not per-patch ResNet-18 windows.
- τ is applied post hoc in batched eval (`stop_index`). `SaccadeViT.infer` stops for real. Both follow the same rule.
- Cost check (random weights, 448² input): B0 2.51 GFLOPs; ours with 2 looks at ρ=0.15 is 1.33 (1.9×), and with 3 looks 1.77.
