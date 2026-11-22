"""Batched evaluation. Every sample takes all K looks once; any stopping threshold tau is then
applied post hoc (identical decisions to `SaccadeViT.infer`, which stops for real at batch 1)."""

import json

import numpy as np
import torch

from ..data import build_loader
from ..models.aggregator import norm_entropy, stop_index
from ..models.saccade import SaccadeViT
from .common import autocast, load_for_eval, out_dir


@torch.no_grad()
def collect(model, loader, cfg, dev, keep_ratio=None, policy=None, k=None) -> dict:
    """-> correct / entropy: (T+1, N) with row 0 = coarse pass (one row for full-image models)."""
    model.eval()
    correct, entropy, labels, area, tokens = [], [], [], [], []
    for batch in loader:
        img, y = batch["image"].to(dev, non_blocking=True), batch["label"].to(dev)
        with autocast(cfg, dev):
            if isinstance(model, SaccadeViT):
                r = model(img, k=k, keep_ratio=keep_ratio, policy=policy, greedy=True,
                          box=batch["box"].to(dev), parts=batch["parts"].to(dev))
                logits, tokens = r.logits, r.tokens
            else:
                logits = model(img).float()[None]
        correct.append((logits.argmax(-1) == y).cpu())
        entropy.append(norm_entropy(logits).cpu())
        labels.append(y.cpu())
        box = batch["box"]
        area.append((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))
    return {
        "correct": torch.cat(correct, 1).numpy(),
        "entropy": torch.cat(entropy, 1).numpy(),
        "labels": torch.cat(labels).numpy(),
        "area": torch.cat(area).numpy(),
        "tokens": np.array(tokens),
    }


def at_tau(res: dict, tau: float, cfg) -> dict:
    """Accuracy and cost when each sample stops at the first look with entropy < tau."""
    looks = stop_index(torch.from_numpy(res["entropy"]), tau).numpy()
    acc = res["correct"][looks, np.arange(len(looks))].mean()
    per_look = res["tokens"][0] if len(res["tokens"]) else 0
    n, depth, cd = 196, 12, cfg.model.get("coarse_depth", 0)
    rel_cost = (cd * n + looks * depth * per_look) / (depth * n)  # token-blocks vs B0 (DESIGN.md fix #5)
    return {"tau": tau, "acc": float(acc), "looks": float(looks.mean()),
            "tokens": float((looks * per_look).mean()), "rel_cost": float(rel_cost.mean())}


def summarize(res: dict, cfg) -> dict:
    out = {f"acc@{t}": float(res["correct"][t].mean()) for t in range(len(res["correct"]))}
    if len(res["correct"]) > 1:
        out["tau"] = [at_tau(res, tau, cfg) for tau in cfg.eval.taus]
    return out


def run(cfg, split: str) -> dict:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = out_dir(cfg)
    model = load_for_eval(cfg, dev)
    loader = build_loader(cfg, split, *model.backbone.norm_stats, batch_size=cfg.eval.batch_size)
    res = collect(model, loader, cfg, dev)
    np.savez(out / f"eval_{split}.npz", **res)
    summary = summarize(res, cfg)
    (out / f"summary_{split}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary
