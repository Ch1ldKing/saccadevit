"""H2 / H3: one checkpoint, swept over keep ratio x tau (x K_max via the per-look rows).

For each keep ratio, tau* is calibrated on val as the largest tau (fewest looks) whose val
accuracy is within `eval.tol` of the fixed-K_max accuracy; test numbers are reported at tau*.
"""

import csv
import json

import numpy as np
import torch

from ..data import build_loader
from .common import load_for_eval, out_dir
from .evaluate import at_tau, collect


def calibrate(res_val: dict, cfg) -> float:
    target = res_val["correct"][-1].mean() - cfg.eval.tol
    ok = [tau for tau in cfg.eval.taus if at_tau(res_val, tau, cfg)["acc"] >= target]
    return max(ok) if ok else 0.0


def run(cfg) -> list[dict]:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = out_dir(cfg)
    model = load_for_eval(cfg, dev)
    stats = model.backbone.norm_stats
    loaders = {s: build_loader(cfg, s, *stats, batch_size=cfg.eval.batch_size) for s in ("val", "test")}
    rows = []
    for rho in cfg.eval.keep_ratios:
        res = {s: collect(model, loaders[s], cfg, dev, keep_ratio=rho) for s in loaders}
        np.savez(out / f"eval_test_rho{rho}.npz", **res["test"])
        tau_star = calibrate(res["val"], cfg)
        for k in range(len(res["test"]["correct"])):
            rows.append({"keep_ratio": rho, "k_max": k, "rule": "fixed", "tau": 0.0, "acc": float(res["test"]["correct"][k].mean()),
                         "looks": k, "rel_cost": (cfg.model.coarse_depth * 196 + k * 12 * res["test"]["tokens"][0]) / (12 * 196)})
        for tau in cfg.eval.taus:
            rows.append({"keep_ratio": rho, "k_max": cfg.model.k_max, "rule": "tau*" if tau == tau_star else "tau",
                         **at_tau(res["test"], tau, cfg)})
    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
        w.writeheader()
        w.writerows({k: r.get(k) for k in w.fieldnames} for r in rows)
    print(json.dumps([r for r in rows if r["rule"] == "tau*"], indent=2))
    return rows
