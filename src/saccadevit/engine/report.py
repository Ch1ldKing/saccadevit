"""Collect output/ into docs/CURRENT_RESULTS.md: run registry, test accuracy (mean ± std over
seeds), tau* operating points, latency, and hypothesis comparisons. Re-run any time."""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

OUT = Path("output")


def _ms(xs: list[float], pct: bool = True) -> str:
    if not xs:
        return "–"
    f = 100 if pct else 1
    m = np.mean(xs) * f
    return f"{m:.2f}" if len(xs) == 1 else f"{m:.2f} ± {np.std(xs) * f:.2f}"


def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _runs() -> dict[str, list[Path]]:
    runs = defaultdict(list)
    for cfg in sorted(OUT.glob("*/seed*/config.yaml")):
        runs[cfg.parent.parent.name].append(cfg.parent)
    return runs


def _status(d: Path, trains: bool) -> str:
    """✓ finished · ▶ running / partial · – not started"""
    if not d.exists():
        return "–"
    if trains:
        return "✓" if (d / "done").exists() else "▶"
    return "✓" if any(d.glob("summary_test.json")) or any(d.glob("sweep.csv")) else "▶"


def _registry(runs) -> str:
    from ..config import EXP_ROOT, load_config

    rows = []
    for p in sorted(EXP_ROOT.rglob("*.yaml")):
        c = load_config(p)
        m, name = c.model, c.name
        trains = not str(c.eval.ckpt).startswith("output/") or str(c.eval.ckpt).startswith(f"output/{name}/")
        what = m.kind if m.kind != "saccade" else f"{m.policy}, q={m.task}, K={m.k_max}, ρ={m.keep_ratio}"
        role = f"train ({c.train.stage})" if trains else f"eval of {Path(c.eval.ckpt).parts[1]}"
        seeds = " ".join(_status(OUT / name / f"seed{s}", trains) for s in range(3))
        val = [max(json.loads(l)["val_acc"] for l in open(d / "metrics.jsonl"))
               for d in runs.get(name, []) if (d / "metrics.jsonl").exists() and (d / "done").exists()]
        rows.append([name, c.data.dataset, what, role, seeds, _ms(val), f"`output/{name}/seed*/`"])
    return _table(["Experiment", "Data", "Model / policy", "Role", "Seeds 0 1 2", "Best val acc %", "Path"], rows)


def _accuracy(runs) -> str:
    rows = []
    for name, dirs in runs.items():
        sums = [json.loads((d / "summary_test.json").read_text()) for d in dirs if (d / "summary_test.json").exists()]
        if not sums:
            continue
        ks = sorted({k for s in sums for k in s if k.startswith("acc@")}, key=lambda k: int(k[4:]))
        cells = {k: _ms([s[k] for s in sums if k in s]) for k in ks}
        npz = dirs[0] / "eval_test.npz"
        tok = int(np.load(npz)["tokens"][0]) if npz.exists() and len(np.load(npz)["tokens"]) else "–"
        row = [name, len(sums)] + [cells.get(f"acc@{i}", "–") for i in range(4)] + [tok]
        rows.append(row)
    return _table(["Experiment", "Seeds", "acc@0 (coarse / full)", "acc@1", "acc@2", "acc@3", "Tokens/look"], rows)


def _sweep(runs) -> str:
    rows = []
    for name, dirs in runs.items():
        for d in dirs:
            p = d / "sweep.csv"
            if p.exists():
                for r in csv.DictReader(open(p)):
                    if r["rule"] == "tau*":
                        rows.append([name, d.name, r["keep_ratio"], r["tau"], f"{100 * float(r['acc']):.2f}",
                                     f"{float(r['looks']):.2f}", f"{float(r['rel_cost']):.3f}"])
    return _table(["Experiment", "Seed", "ρ", "τ*", "Test acc %", "Mean looks", "Rel. cost (token-blocks)"], rows) if rows else "_no sweeps yet_"


def _latency(runs) -> str:
    rows = []
    for name, dirs in runs.items():
        for p in sorted(dirs[0].glob("latency_*.json")):
            r = json.loads(p.read_text())
            sec = ", ".join(f"{k} {v:.2f}" for k, v in r["sections_ms_per_img"].items())
            mem = f"{r['peak_mem_mb']:.0f}" if r["peak_mem_mb"] else "–"
            rows.append([name, r["device"], r["keep_ratio"], r["tau"], f"{r['p50_ms']:.2f}", f"{r['p95_ms']:.2f}",
                         f"{r['mean_looks']:.2f}", f"{r['gflops_per_img']:.3f}", mem, sec])
    return _table(["Experiment", "Device", "ρ", "τ", "p50 ms", "p95 ms", "Looks", "GFLOPs", "Peak MB", "Per-module ms (profiled)"], rows) if rows else "_no latency runs yet_"


def _compare() -> str:
    rows = []
    for p in sorted((OUT / "compare").glob("*.json")):
        r = json.loads(p.read_text())
        a = r["all"]
        strata = ", ".join(f"{k} {100 * r[k]['diff']:+.2f}" for k in ("coarse_wrong", "small", "large") if k in r)
        rows.append([p.stem, f"{r['rule_a']} → {r['rule_b']}", f"{100 * r['acc_a']:.2f} → {100 * r['acc_b']:.2f}",
                     f"{100 * a['diff']:+.2f} [{100 * a['lo']:+.2f}, {100 * a['hi']:+.2f}]", f"{a['p_le_0']:.3f}", strata])
    return _table(["Comparison (B − A)", "Rules", "Acc %", "Δ pt [95% CI]", "P(Δ ≤ 0)", "Δ by stratum"], rows) if rows else "_no comparisons yet_"


def write(path: str = "docs/CURRENT_RESULTS.md") -> None:
    runs = _runs()
    text = f"""# Current results

Generated by `make report` at {datetime.now():%Y-%m-%d %H:%M} from `output/`. Do not edit by hand.
Platform for these numbers: see the Device column (preliminary runs on RTX 3090, final latency on L40).
Accuracy is on the test split; mean ± std over the seeds listed. acc@k = after k looks, with acc@0 = coarse pass (or the full model for B0/B4).

## Runs

{_registry(runs)}

## Test accuracy (%)

{_accuracy(runs)}

## Operating points (τ* calibrated on val, reported on test)

{_sweep(runs)}

## Latency (batch 1)

{_latency(runs)}

## Hypothesis tests (paired bootstrap over test images, seeds pooled)

{_compare()}
"""
    Path(path).write_text(text)
    print(f"wrote {path} ({len(runs)} experiments)")
