# Third-party reference implementations

These are read-only snapshots at pinned commits, produced by `make third_party_fetch` from `configs/assets.yaml`. Figures, notebooks and checkpoints are stripped. Each folder keeps its upstream `LICENSE` and a `SNAPSHOT` file recording the repo, commit and licence.

**Nothing here is imported at runtime.** `src/saccadevit` reimplements what it needs. The snapshots exist so the reimplementations can be audited line by line against upstream. Runtime dependencies (timm's ViT and so on) are pinned in `uv.lock` instead.

| Directory | Upstream @ commit | Licence | What we compare against |
|---|---|---|---|
| `adaptivenn` | LeapLabTHU/AdaptiveNN @ `020bf8d` | MIT | Coarse-to-fine loop, confidence stopping, self-rewarding PPO (our M2/M4 and Stage B, `engine_ppo.py`) |
| `dynamicvit` | raoyongming/DynamicViT @ `1322e62` | MIT | One-pass input-adaptive token pruning (the prior work behind Q2) |
| `ptopk` | google-research/google-research `ptopk_patch_selection` @ `20c6bc7` | Apache-2.0 | Perturbed top-K (`lib/`). Our `PerturbedTopK` in `models/glimpse.py` |

Not snapshotted: FocusViT and CAIT publish no code (B3 and B4 are our reimplementations from the papers). Dynamic-Vision-Transformer has no licence, so it is cited only.
