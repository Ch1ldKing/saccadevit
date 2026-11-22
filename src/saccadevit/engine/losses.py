"""One loss per training stage. Each returns (loss, logs)."""

import random

import torch
import torch.nn.functional as F


def _ce(logits, y, cfg):
    return F.cross_entropy(logits, y, label_smoothing=cfg.train.label_smoothing)


def loss_full(model, batch, cfg):
    """Stage 0 (B0) and B4: plain classification on the whole image."""
    logits = model(batch["image"])
    return _ce(logits, batch["label"], cfg), {"acc": (logits.argmax(-1) == batch["label"]).float().mean()}


def loss_stage_a(model, batch, cfg):
    """M3 + M4 with teacher fixations (gt / random / full view), random K and keep ratio
    so one checkpoint serves every budget (DESIGN.md fixes #4, #6)."""
    k = random.randint(1, cfg.model.k_max)
    rho = random.uniform(*cfg.train.keep_ratio_range)
    y = batch["label"]
    r = model(batch["image"], k=k, keep_ratio=rho, policy="mix", box=batch["box"], parts=batch["parts"])
    looks = torch.stack([_ce(l, y, cfg) for l in r.logits[1:]]).mean()
    loss = looks + cfg.train.coarse_weight * _ce(r.logits[0], y, cfg)
    return loss, {"ce_looks": looks, "acc": (r.logits[-1].argmax(-1) == y).float().mean()}


def loss_stage_b(model, batch, cfg):
    """REINFORCE with a self-reward: r_t = log p_t(y) - log p_{t-1}(y) - lambda * tokens_t / N.
    No fixation labels. A per-step batch-mean baseline; CE keeps M3/M4 calibrated if they train."""
    t = cfg.train
    rho = random.uniform(*t.keep_ratio_range)
    y = batch["label"]
    r = model(batch["image"], keep_ratio=rho, policy="learned", greedy=False)
    logp_y = r.logits.log_softmax(-1).gather(-1, y[None, :, None].expand(len(r.logits), -1, 1)).squeeze(-1)
    n = model.backbone.num_patches
    cost = torch.tensor(r.tokens, device=y.device, dtype=torch.float)[:, None] / n
    reward = (logp_y[1:] - logp_y[:-1]).detach() - t.cost_weight * cost  # (T, B)
    ret = reward.flip(0).cumsum(0).flip(0)
    adv = (ret - ret.mean(1, keepdim=True)) / (ret.std(1, keepdim=True) + 1e-6)
    pg = -(adv * r.logp).mean()
    ent = r.entropy.mean()
    ce = torch.stack([_ce(l, y, cfg) for l in r.logits[1:]]).mean()
    loss = pg - t.entropy_weight * ent + t.ce_weight * ce
    return loss, {"pg": pg, "ent": ent, "ret": ret[0].mean(), "acc": (r.logits[-1].argmax(-1) == y).float().mean()}


LOSSES = {"full": loss_full, "a": loss_stage_a, "b": loss_stage_b}
