"""SaccadeViT: the slide-13 loop on one shared backbone.

    C, g, p0 = coarse(img);  q = task(p0);  e = init(g)
    for t < K:  a = policy(C, g, e, q) -> glimpse(img, a, rho) -> z;  e, p = agg(e, z);  stop if H(p) < tau
"""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn

from .aggregator import Aggregator, norm_entropy
from .backbone import Backbone
from .coarse import CoarseLook
from .glimpse import Glimpse, action_boxes
from .policy import (
    Action,
    LearnedPolicy,
    TaskEncoder,
    covered_cells,
    full_view_action,
    gt_action,
    random_action,
    saliency_action,
    spectral_residual,
)


def _no_timer(_name):
    return nullcontext()


@dataclass
class State:
    C: torch.Tensor
    g: torch.Tensor
    q: torch.Tensor
    e: torch.Tensor
    p0: torch.Tensor
    visited: torch.Tensor
    saliency: torch.Tensor | None = None
    aux: float | torch.Tensor = 0.0  # scorer distillation loss summed over looks


@dataclass
class Rollout:
    logits: torch.Tensor  # (T+1, B, C); row 0 = coarse pass
    logp: torch.Tensor | None  # (T, B), learned policy only
    entropy: torch.Tensor | None  # (T, B)
    tokens: list[int]  # tokens fed to the ViT per look
    aux: float | torch.Tensor = 0.0  # mean scorer distillation loss per look


class SaccadeViT(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.backbone = bb = Backbone(m.backbone, m.pretrained, m.num_classes, m.weights)
        D, self.grid = bb.dim, bb.grid
        self.coarse = CoarseLook(D, m.coarse_depth, m.num_classes)
        self.task = TaskEncoder(m.task, m.num_classes, D)
        self.policy = LearnedPolicy(D, len(m.scales))
        self.glimpse = Glimpse(D, self.grid, m.translate, m.offset_pe, m.topk_sigma, m.topk_samples, m.select_distill, m.fuse_rest, m.select)
        self.agg = Aggregator(D, m.num_classes)
        self.img_size = m.img_size
        self.num_scales = len(m.scales)
        # side of each scale as a fraction of the image; last entry = whole image
        self.register_buffer("sides", torch.tensor([*m.scales, m.img_size], dtype=torch.float) / m.img_size, persistent=False)
        self.m = m

    # -- loop pieces -------------------------------------------------------------------
    def begin(self, img) -> State:
        C, g, p0 = self.coarse(img, self.backbone)
        visited = torch.zeros(C.shape[:2], dtype=torch.bool, device=C.device)
        return State(C, g, self.task(p0), self.agg.start(g), p0, visited)

    def act(self, st: State, t: int, policy: str, greedy: bool, img, box=None, parts=None) -> Action:
        B, dev = len(img), img.device
        if policy == "learned":
            return self.policy(st.C, torch.cat([st.g, st.e, st.q], -1), st.visited, greedy)
        if policy == "random":
            return random_action(st.visited, self.num_scales)
        if policy == "center":
            return full_view_action(B, self.grid, self.num_scales, dev)
        if policy == "saliency":
            if st.saliency is None:
                st.saliency = spectral_residual(img, self.grid)
            return saliency_action(st.saliency, st.visited, self.m.saliency_scale)
        if policy == "gt":
            return gt_action(box, parts, t, self.sides[:-1] * self.img_size, self.img_size, self.grid)
        if policy == "mix":  # Stage-A teacher: per-sample draw of gt / random / full view
            options = [self.act(st, t, p, greedy, img, box, parts) for p in ("gt", "random", "center")]
            pick = torch.multinomial(torch.tensor(list(self.m.mix_probs), device=dev).expand(B, -1), 1).squeeze(1)
            cell = torch.stack([o.cell for o in options], 1).gather(1, pick[:, None]).squeeze(1)
            scale = torch.stack([o.scale for o in options], 1).gather(1, pick[:, None]).squeeze(1)
            return Action(cell, scale)
        raise ValueError(policy)

    def aim(self, st: State, a: Action) -> torch.Tensor:
        """Box of an action; marks its cells visited (inhibition of return)."""
        box = action_boxes(a.cell, self.sides[a.scale], self.grid)
        st.visited = st.visited | covered_cells(box, a.cell, self.grid)
        return box

    def look(self, st: State, img, a: Action, keep_ratio: float, sec=_no_timer):
        z, k, aux = self.glimpse(img, self.aim(st, a), keep_ratio, self.backbone, sec)
        st.aux = st.aux + aux
        with sec("aggregate"):
            st.e, logits = self.agg.step(st.e, z)
        return logits, k

    def look_parallel(self, st: State, img, acts: list[Action], keep_ratio: float, sec=_no_timer):
        """F3: K boxes chosen up front -> one batched ViT pass over all K crops -> GRU in order."""
        B, k = len(img), len(acts)
        boxes = torch.cat([self.aim_box(a) for a in acts])  # (K*B, 4), look-major
        idx = torch.arange(B, device=img.device).repeat(k)
        z, n, aux = self.glimpse(img, boxes, keep_ratio, self.backbone, sec, img_idx=idx)
        st.aux = st.aux + aux * k
        out = []
        with sec("aggregate"):
            for zt in z.view(k, B, -1):
                st.e, logits = self.agg.step(st.e, zt)
                out.append(logits)
        return out, n

    def aim_box(self, a: Action) -> torch.Tensor:
        return action_boxes(a.cell, self.sides[a.scale], self.grid)

    def plan(self, st: State, k: int, policy: str, greedy: bool, img, box=None, parts=None) -> list[Action]:
        """F3: all K fixations from the coarse map alone (e stays e_0), with inhibition of return."""
        acts = []
        for t in range(k):
            a = self.act(st, t, policy, greedy, img, box, parts)
            self.aim(st, a)
            acts.append(a)
        return acts

    # -- entry points ------------------------------------------------------------------
    def forward(self, img, k=None, keep_ratio=None, policy=None, greedy=True, box=None, parts=None) -> Rollout:
        """Exactly k looks for the whole batch (training and batched evaluation)."""
        k = self.m.k_max if k is None else k
        keep_ratio = self.m.keep_ratio if keep_ratio is None else keep_ratio
        policy = policy or self.m.policy
        st = self.begin(img)
        logits, tokens = [st.p0], []
        if self.m.parallel and k > 0:
            acts = self.plan(st, k, policy, greedy, img, box, parts)
            outs, n = self.look_parallel(st, img, acts, keep_ratio)
            logits += outs
            tokens = [n] * k
        else:
            acts = []
            for t in range(k):
                a = self.act(st, t, policy, greedy, img, box, parts)
                out, n = self.look(st, img, a, keep_ratio)
                logits.append(out)
                tokens.append(n)
                acts.append(a)
        stack = lambda xs: torch.stack(xs) if xs else None  # noqa: E731
        logps = [a.logp for a in acts if a.logp is not None]
        ents = [a.entropy for a in acts if a.entropy is not None]
        return Rollout(torch.stack([x.float() for x in logits]), stack(logps), stack(ents), tokens, st.aux / max(k, 1))

    @torch.no_grad()
    def infer(self, img, tau: float, k=None, keep_ratio=None, policy=None, timer=None):
        """Batch-1 inference with real early stopping. Returns (logits, looks).
        Parallel mode checks tau once, after the coarse pass."""
        sec = timer.section if timer else _no_timer
        k = self.m.k_max if k is None else k
        keep_ratio = self.m.keep_ratio if keep_ratio is None else keep_ratio
        policy = policy or self.m.policy
        with sec("coarse"):
            st = self.begin(img)
        logits = st.p0
        if tau > 0 and norm_entropy(logits).item() < tau:
            return logits, 0
        if self.m.parallel and k > 0:
            with sec("policy"):
                acts = self.plan(st, k, policy, True, img)
            return self.look_parallel(st, img, acts, keep_ratio, sec)[0][-1], k
        for t in range(k):
            if t > 0 and tau > 0 and norm_entropy(logits).item() < tau:
                return logits, t
            with sec("policy"):
                a = self.act(st, t, policy, True, img)
            logits, _ = self.look(st, img, a, keep_ratio, sec)
        return logits, k

    # -- CUDA-graph pieces (F1): tensors in, tensors out, no host sync ------------------
    def g_begin(self, img):
        st = self.begin(img)
        return st.C, st.g, st.q, st.e, st.visited, st.p0.float()

    def g_step(self, img, C, g, q, e, visited, keep_ratio: float):
        st = State(C, g, q, e, None, visited)
        logits, _ = self.look(st, img, self.act(st, 0, self.m.policy, True, img), keep_ratio)
        return st.e, st.visited, logits.float()

    def g_all(self, img, k: int, keep_ratio: float):
        """Fixed-k greedy rollout (tau = 0) as one function -> final logits."""
        return self.infer(img, 0.0, k=k, keep_ratio=keep_ratio)[0].float()
