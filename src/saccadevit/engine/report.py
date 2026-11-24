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
            mode = ("graph" if r.get("graph") else "eager") + (f" b{r['batch']}" if r.get("batch", 1) > 1 else "")
            rows.append([name, r["device"], mode, r["keep_ratio"], r.get("k_max", "–"), r["tau"], f"{r['p50_ms']:.2f}",
                         f"{r['p95_ms']:.2f}", f"{r.get('img_per_s', 1e3 / r['p50_ms']):.0f}", f"{r['mean_looks']:.2f}",
                         f"{r['gflops_per_img']:.3f}", mem, sec])
    head = ["Experiment", "Device", "Mode", "ρ", "K", "τ", "p50 ms", "p95 ms", "img/s", "Looks", "GFLOPs", "Peak MB", "Per-module ms (eager)"]
    return _table(head, rows) if rows else "_no latency runs yet_"


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _pareto(prefix: str) -> str:
    """T2-T4: accuracy (test, mean over seeds) vs batch-1 GPU ms with CUDA graphs (seed 0), Pareto set marked."""
    pts = []  # (label, tokens, acc %, gpu ms, img/s at b64). Two fronts: batch-1 ms and batch-64 throughput
    for name, label, key in [("b0_full", "B0 224", "acc@0"), ("b0_448", "B0 native", "acc@0"), ("b0_native", "B0 native", "acc@0"),
                             ("b4_merge", "B4 merge", "acc@0"), ("b3_focusvit", "B3 FocusViT-style", "acc@1")]:
        accs = [r[key] for d in sorted(OUT.glob(f"{prefix}{name}/seed*")) if (r := _load(d / "summary_test.json")) and key in r]
        lat = _load(OUT / f"{prefix}{name}/seed0/latency_cuda_graph.json")
        if accs and lat:
            thr = _load(OUT / f"{prefix}{name}/seed0/latency_cuda_graph_b64.json")
            pts.append((label, "", 100 * np.mean(accs), lat["p50_ms"], thr and thr["img_per_s"]))
    for sweep, run, label in [("a3_keep_sweep", "b5_ours", "B5 ours"), ("a3_keep_sweep_par", "b5_par", "B5-par ours"),
                              ("keep_sweep", "b5_ours", "B5 ours"), ("keep_sweep_par", "b5_par", "B5-par ours")]:
        acc = defaultdict(list)
        for f in sorted(OUT.glob(f"{prefix}{sweep}/seed*/sweep.csv")):
            for r in csv.DictReader(open(f)):
                if r["rule"] == "fixed" and int(r["k_max"]) > 0:
                    acc[(float(r["keep_ratio"]), int(r["k_max"]))].append(float(r["acc"]))
        for (rho, k), a in sorted(acc.items()):
            lat = _load(OUT / f"{prefix}{run}/seed0/latency_cuda_graph_rho{rho}_k{k}.json")
            if lat:
                thr = _load(OUT / f"{prefix}{run}/seed0/latency_cuda_graph_b64_rho{rho}_k{k}.json")
                g = max(1, round((rho * 196) ** 0.5))
                pts.append((f"{label} ρ={rho} K={k}", f"{k}×{g * g}", 100 * np.mean(a), lat["p50_ms"], thr and thr["img_per_s"]))
    if not pts:
        return "_no trade-off runs yet_"
    front_ms = lambda p: not any(q[2] >= p[2] and q[3] <= p[3] and (q[2] > p[2] or q[3] < p[3]) for q in pts)  # noqa: E731
    thr = [q for q in pts if q[4]]
    front_thr = lambda p: p[4] is not None and not any(q[2] >= p[2] and q[4] >= p[4] and (q[2] > p[2] or q[4] > p[4]) for q in thr)  # noqa: E731
    fmt = lambda v, f: "–" if v is None else format(v, f)  # noqa: E731
    rows = [[("**" + p[0] + "**") if front_ms(p) or front_thr(p) else p[0], p[1], f"{p[2]:.2f}", f"{p[3]:.2f}", fmt(p[4], ".0f"),
             "✓" if front_ms(p) else "", "✓" if front_thr(p) else ""] for p in sorted(pts, key=lambda p: -p[2])]
    return _table(["Model / operating point", "Tokens", "Test acc %", "GPU ms (b1, graph)", "img/s (b64, graph)", "Pareto ms", "Pareto img/s"], rows)


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
Platform: see the Device column (phase 1 latency on RTX 3090 hitcvsz4, phase 2 on RTX 5090 D newpc, bf16 autocast).
Accuracy is on the test split; mean ± std over the seeds listed. acc@k = after k looks, with acc@0 = coarse pass (or the full model for B0/B4).

## Runs

{_registry(runs)}

## Test accuracy (%)

{_accuracy(runs)}

## Operating points (τ* calibrated on val, reported on test)

{_sweep(runs)}

## Latency (batch 1)

{_latency(runs)}

## Accuracy vs time: CUB (Pareto set in bold)

{_pareto("")}

## Accuracy vs time: canvas

{_pareto("canvas-")}

## Hypothesis tests (paired bootstrap over test images, seeds pooled)

{_compare()}
"""
    Path(path).write_text(text)
    print(f"wrote {path} ({len(runs)} experiments)")
