# SaccadeViT

**Look · keep · stop on one ~5.7M-parameter ViT.**

- **Look:** a coarse look proposes, and a task-conditioned policy aims a crop.
- **Keep:** a FocusViT-style glimpse keeps the top-K translated patches.
- **Stop:** a GRU accumulates evidence until the prediction is confident.

The speed-up is measured in batch-1 milliseconds, not FLOPs.

```bash
make setup && make input_verify && make test
make help
make preliminary
```

| Doc | Contents |
|---|---|
| [docs/MODEL.md](docs/MODEL.md) | What the method does and its optimisation objectives per stage |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Setup, make targets, dependencies between runs, outputs |
| [third_party/README.md](third_party/README.md) | Pinned reference implementations |
| `../DESIGN.md`, `../DECISIONS.md` | Design rationale and the open decisions (outside this repo) |

## Layout

```
Makefile                 every experiment as an underscore target
configs/base.yaml        shared defaults;  configs/exp/**.yaml  one file per experiment (_base_ inheritance)
configs/assets.yaml      datasets, weights, third-party commits
src/saccadevit/
  data/                  cub (boxes, 15 parts), canvas (synthetic 1024² search), cifar (control)
  models/                backbone (shared ViT, B0) · coarse (M1) · policy (M2) · glimpse (M3)
                         · aggregator (M4) · saccade (the loop) · merging (B4)
  engine/                losses · train · evaluate · sweep · latency · stats
  assets.py validate.py cli.py config.py
input/  output/          git-ignored: datasets + weights / runs
third_party/             committed snapshots, not imported
```
