"""M3 - look (after FocusViT): crop -> translate patches -> offset/location-aware PE -> top-K.

Differences from FocusViT, all deliberate (DESIGN.md fixes #6, #7):
  * the translator is a light strided conv over the crop (one pass, stride 16), not
    ResNet-18 layers per patch window;
  * every patch also carries its absolute location in the full image (`loc_pe`);
  * keep ratio is a call argument, so one model serves many budgets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .backbone import VIT_RES, Backbone


def mlp(i: int, h: int, o: int, zero_last: bool = True) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))
    if zero_last:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


def cell_centers(grid: int, device) -> torch.Tensor:  # (N, 2) xy in [0, 1]
    c = (torch.arange(grid, device=device) + 0.5) / grid
    y, x = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([x.flatten(), y.flatten()], 1)


def action_boxes(cell, side, grid: int) -> torch.Tensor:
    """cell: (B,) index on the coarse grid; side: (B,) normalised crop side -> (B, 4) xyxy kept inside [0, 1]."""
    half = (side / 2).clamp(max=0.5)
    cx = ((cell % grid + 0.5) / grid).clamp(half, 1 - half)
    cy = ((cell // grid + 0.5) / grid).clamp(half, 1 - half)
    return torch.stack([cx - half, cy - half, cx + half, cy + half], 1)


class PerturbedTopK(torch.autograd.Function):
    """Differentiable top-K via perturbed maximisers (Berthet et al. 2020; Cordonnier et al. 2021)."""

    @staticmethod
    def forward(ctx, scores, k: int, samples: int, sigma: float):
        B, N = scores.shape
        noise = torch.randn(B, samples, N, device=scores.device, dtype=scores.dtype)
        idx = (scores[:, None] + sigma * noise).topk(k, dim=-1).indices.sort(dim=-1).values
        onehot = F.one_hot(idx, N).to(scores.dtype)  # (B, S, k, N)
        ctx.save_for_backward(onehot, noise)
        ctx.sigma = sigma
        return onehot.mean(1)  # (B, k, N)

    @staticmethod
    def backward(ctx, grad):
        onehot, noise = ctx.saved_tensors  # d indicators / d scores = E[onehot * noise] / sigma
        a = torch.einsum("bkn,bskn->bs", grad, onehot)
        return torch.einsum("bs,bsm->bm", a, noise) / (onehot.shape[1] * ctx.sigma), None, None, None


class TopK(nn.Module):
    def __init__(self, sigma: float, samples: int):
        super().__init__()
        self.sigma, self.samples = sigma, samples

    def forward(self, x, scores, k: int):
        if self.training:
            s = (scores - scores.mean(1, keepdim=True)) / (scores.std(1, keepdim=True) + 1e-6)
            return PerturbedTopK.apply(s.float(), k, self.samples, self.sigma).to(x.dtype) @ x
        idx = scores.topk(k, dim=1).indices.sort(dim=1).values
        return x.gather(1, idx[..., None].expand(-1, -1, x.shape[-1]))


class Translator(nn.Module):
    """Per-patch (dx, dy) in patch units; receptive field ~ a 32x32 window around each patch."""

    def __init__(self, max_shift: float = 0.5, width: int = 32):
        super().__init__()
        w = width
        self.max_shift = max_shift
        self.net = nn.Sequential(
            nn.Conv2d(3, w, 4, 4), nn.GELU(),
            nn.Conv2d(w, w, 3, 2, 1), nn.GELU(),
            nn.Conv2d(w, w, 3, 2, 1), nn.GELU(),
            nn.Conv2d(w, 2, 3, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, crop):  # (B, 2, g, g)
        return torch.tanh(self.net(crop)) * self.max_shift


def shift_patches(crop: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Resample each patch of `crop` displaced by its own offset (patch units)."""
    B, _, S, _ = crop.shape
    g = offset.shape[-1]
    eye = torch.eye(2, 3, device=crop.device, dtype=crop.dtype).expand(B, -1, -1)
    base = F.affine_grid(eye, (B, 3, S, S), align_corners=False)
    shift = F.interpolate(offset, size=S, mode="nearest").permute(0, 2, 3, 1) * (2 / g)
    return F.grid_sample(crop, base + shift, mode="bilinear", padding_mode="border", align_corners=False)


class Glimpse(nn.Module):
    def __init__(self, dim: int, grid: int, translate: bool, offset_pe: bool, sigma: float, samples: int):
        super().__init__()
        self.grid = grid
        self.translator = Translator() if translate else None
        self.offset_pe = mlp(2, 64, dim) if translate and offset_pe else None
        self.loc_pe = mlp(3, 64, dim)
        self.scorer = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.topk = TopK(sigma, samples)

    def forward(self, img, box, keep_ratio: float, bb: Backbone, sec):
        """img: (B, 3, S, S); box: (B, 4) normalised xyxy -> (z: (B, D), tokens kept)."""
        with sec("crop"):
            S = img.shape[-1]
            rois = torch.cat([torch.arange(len(box), device=box.device, dtype=box.dtype)[:, None], box * S], 1)
            crop = roi_align(img, rois.to(img.dtype), VIT_RES, aligned=True, sampling_ratio=2)
            if self.translator is not None:
                off = self.translator(crop)
                crop = shift_patches(crop, off)
        with sec("embed"):
            x = bb.patchify(crop) + bb.pos_patch
            if self.offset_pe is not None:
                x = x + self.offset_pe(off.flatten(2).transpose(1, 2))
            side = (box[:, 2] - box[:, 0])[:, None, None]
            xy = box[:, None, :2] + cell_centers(self.grid, box.device)[None] * side
            x = x + self.loc_pe(torch.cat([xy, side.expand(-1, xy.shape[1], 1)], -1).to(x.dtype))
            k = max(1, round(keep_ratio * x.shape[1]))
            if k < x.shape[1]:
                x = self.topk(x, self.scorer(x).squeeze(-1), k)
        with sec("vit"):
            z = bb.vit.norm(bb.encode(x))[:, 0]
        return z, k
