"""Generic trainer; the stage decides the loss and which parameters move."""

import json
import math
import time

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..data import build_loader
from ..models import build_model, load_weights
from .common import autocast, device, out_dir, seed_all, to_device
from .evaluate import collect
from .losses import LOSSES


def param_groups(model, cfg) -> list[dict]:
    """full: everything. a: new modules + last blocks of the backbone. b: policy (+ the rest, scaled)."""
    t, lr = cfg.train, cfg.train.lr
    if t.stage == "full":
        spec = [(model, lr)]
    else:
        vit = model.backbone.vit
        tuned = [*vit.blocks[len(vit.blocks) - t.tune_last_blocks:], vit.norm] if t.tune_last_blocks else []
        heads = [model.coarse, model.task, model.glimpse, model.agg]
        if t.stage == "a":
            spec = [*((m, lr) for m in heads), *((m, lr * t.backbone_lr_scale) for m in tuned)]
        else:
            joint = lr * t.joint_lr_scale
            spec = [(model.policy, lr), *((m, joint) for m in heads), *((m, joint * t.backbone_lr_scale) for m in tuned)]
    model.requires_grad_(False)
    groups = []
    for module, group_lr in spec:
        if group_lr > 0:
            module.requires_grad_(True)
            groups.append({"params": list(module.parameters()), "lr": group_lr})
    return groups


def cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def validate(model, loader, cfg, dev) -> float:
    res = collect(model, loader, cfg, dev)
    return float(res["correct"][-1].mean())


def run(cfg) -> None:
    seed_all(cfg.seed)
    dev = device()
    out = out_dir(cfg)
    model = build_model(cfg.model)
    if cfg.train.init:
        load_weights(model, cfg.train.init)
    model.to(dev)
    mean, std = model.backbone.norm_stats
    train_loader = build_loader(cfg, "train", mean, std)
    val_loader = build_loader(cfg, "val", mean, std)

    groups = param_groups(model, cfg)
    opt = torch.optim.AdamW(groups, weight_decay=cfg.train.weight_decay)
    total = cfg.train.epochs * len(train_loader)
    warmup = cfg.train.warmup_epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine(s, total, warmup))
    loss_fn = LOSSES[cfg.train.stage]
    print(OmegaConf.to_yaml(cfg))
    print(f"trainable params: {sum(p.numel() for g in groups for p in g['params']) / 1e6:.2f}M")

    best, log = -1.0, open(out / "metrics.jsonl", "a")
    for epoch in range(cfg.train.epochs):
        model.train()
        t0, running = time.time(), {}
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}", leave=False)):
            batch = to_device(batch, dev)
            with autocast(cfg, dev):
                loss, logs = loss_fn(model, batch, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], cfg.train.grad_clip)
            opt.step()
            sched.step()
            for k, v in {"loss": loss, **logs}.items():
                running[k] = running.get(k, 0.0) + v.detach()
        val_acc = validate(model, val_loader, cfg, dev)
        rec = {"epoch": epoch, "val_acc": val_acc, "time": time.time() - t0, **{k: float(v) / (step + 1) for k, v in running.items()}}
        print(json.dumps(rec))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        ckpt = {"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg), "epoch": epoch, "val_acc": val_acc}
        torch.save(ckpt, out / "last.pt")
        if val_acc > best:
            best = val_acc
            torch.save(ckpt, out / "best.pt")
    print(f"best val acc {best:.4f} -> {out / 'best.pt'}")
