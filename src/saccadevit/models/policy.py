"""M2 - aim (our addition 1) and the heuristic policies it is compared against.

Every policy picks an Action on the same discrete space: one cell of the coarse 14x14 grid
plus a scale index (DESIGN.md fix #8). Scale index == num_scales means "the whole image"
(used by the centre / FocusViT-style one-look baseline).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .glimpse import cell_centers, mlp


@dataclass
class Action:
    cell: torch.Tensor  # (B,) long
    scale: torch.Tensor  # (B,) long
    logp: torch.Tensor | None = None
    entropy: torch.Tensor | None = None


class TaskEncoder(nn.Module):
    """q. 'coarse_logits': embedding of the coarse class belief (top-down). 'none': zeros (A5)."""

    def __init__(self, kind: str, num_classes: int, dim: int):
        super().__init__()
        self.kind, self.dim = kind, dim
        self.proj = nn.Linear(num_classes, dim)

    def forward(self, p0):
        if self.kind == "none":
            return p0.new_zeros(len(p0), self.dim)
        return self.proj(p0.detach().float().softmax(-1))


class LearnedPolicy(nn.Module):
    """pi(a | C, g, e, q): a query from [g, e, q] attends over the coarse map C."""

    def __init__(self, dim: int, num_scales: int, hidden: int = 128):
        super().__init__()
        self.hidden = hidden
        self.query = mlp(3 * dim, hidden, hidden, zero_last=False)
        self.key = nn.Linear(dim, hidden)
        self.scale = mlp(3 * dim, hidden, num_scales, zero_last=False)

    def forward(self, C, ctx, visited, greedy: bool) -> Action:
        C, ctx = C.detach().float(), ctx.detach().float()
        logits = torch.einsum("bnh,bh->bn", self.key(C), self.query(ctx)).float() / math.sqrt(self.hidden)
        cells = Categorical(logits=logits.masked_fill(visited, -1e4))
        scales = Categorical(logits=self.scale(ctx).float())
        if greedy:
            cell, scale = cells.probs.argmax(-1), scales.probs.argmax(-1)
        else:
            cell, scale = cells.sample(), scales.sample()
        return Action(cell, scale, cells.log_prob(cell) + scales.log_prob(scale), cells.entropy() + scales.entropy())


def random_action(visited, num_scales: int) -> Action:
    cell = torch.multinomial((~visited).float() + 1e-6, 1).squeeze(1)
    return Action(cell, torch.randint(num_scales, cell.shape, device=cell.device))


def full_view_action(B: int, grid: int, num_scales: int, device) -> Action:
    centre = torch.full((B,), (grid // 2) * grid + grid // 2, device=device)
    return Action(centre, torch.full((B,), num_scales, device=device))


def spectral_residual(img: torch.Tensor, grid: int) -> torch.Tensor:
    """Bottom-up saliency (Hou & Zhang 2007) pooled to the coarse grid -> (B, N)."""
    x = F.interpolate(img.float().mean(1, keepdim=True), size=64, mode="area")
    f = torch.fft.fft2(x)
    log_amp, phase = torch.log(f.abs() + 1e-8), torch.angle(f)
    resid = log_amp - F.avg_pool2d(log_amp, 3, 1, 1, count_include_pad=False)
    sal = torch.fft.ifft2(torch.exp(torch.complex(resid, phase))).abs() ** 2
    sal = F.avg_pool2d(F.avg_pool2d(sal, 5, 1, 2), 5, 1, 2)
    return F.adaptive_avg_pool2d(sal, grid).flatten(1)


def saliency_action(sal, visited, scale: int) -> Action:
    cell = sal.masked_fill(visited, -1).argmax(-1)
    return Action(cell, torch.full_like(cell, scale))


def gt_action(box, parts, step: int, scales_px, img_size: int, grid: int) -> Action:
    """Oracle / Stage-A teacher. Step 0: the object box at the smallest scale covering it.
    Later steps: a random visible part at a random scale (falls back to the box)."""
    B, dev = len(box), box.device
    centre = (box[:, :2] + box[:, 2:]) / 2
    if step == 0:
        side = (box[:, 2:] - box[:, :2]).max(1).values * img_size
        scale = torch.searchsorted(scales_px.to(dev).float(), side.contiguous()).clamp(max=len(scales_px) - 1)
    else:
        vis = parts[..., 2] > 0
        pick = torch.multinomial(vis.float() + 1e-6, 1).squeeze(1)
        part = parts[torch.arange(B, device=dev), pick, :2]
        centre = torch.where(vis.any(1, keepdim=True), part, centre)
        scale = torch.randint(len(scales_px), (B,), device=dev)
    ij = (centre * grid).long().clamp(0, grid - 1)
    return Action(ij[:, 1] * grid + ij[:, 0], scale)


def covered_cells(box, cell, grid: int) -> torch.Tensor:
    """Inhibition of return: cells whose centre lies in the central half of the crop, plus the aimed cell."""
    c = cell_centers(grid, box.device)[None]  # (1, N, 2)
    mid, half = (box[:, None, :2] + box[:, None, 2:]) / 2, (box[:, None, 2:] - box[:, None, :2]) / 4
    inside = ((c - mid).abs() <= half).all(-1)
    return inside | F.one_hot(cell, grid * grid).bool()
