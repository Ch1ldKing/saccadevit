"""B4: CAIT-style asymmetric neighbour merging (ATME) on the full image.

Two neighbours are concatenated -> LayerNorm -> Linear back to D, so the grid halves but
stays a grid: 196 -> 98 (merge along width) -> 49 (merge along height). The projection is
zero-initialised on top of the neighbour mean, so a B0 checkpoint starts as average pooling.
"""

import torch
import torch.nn as nn

from .backbone import Backbone, to_vit_res


class NeighbourMerge(nn.Module):
    def __init__(self, dim: int, axis: str):
        super().__init__()
        self.axis = axis
        self.norm = nn.LayerNorm(2 * dim)
        self.proj = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, hw: tuple[int, int]):
        cls, x = h[:, :1], h[:, 1:]
        B, _, D = x.shape
        H, W = hw
        x = x.reshape(B, H, W, D)
        if self.axis == "w":
            x, hw = x.reshape(B, H, W // 2, 2 * D), (H, W // 2)
        else:
            x, hw = x.reshape(B, H // 2, 2, W, D).transpose(2, 3).reshape(B, H // 2, W, 2 * D), (H // 2, W)
        x = self.proj(self.norm(x)) + 0.5 * (x[..., :D] + x[..., D:])
        return torch.cat([cls, x.flatten(1, 2)], 1), hw


class MergeViT(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.backbone = Backbone(m.backbone, m.pretrained, m.num_classes, m.weights)
        D = self.backbone.dim
        self.merge_at = list(m.merge_at)
        self.merges = nn.ModuleList([NeighbourMerge(D, "w"), NeighbourMerge(D, "h")])

    def forward(self, img):
        bb = self.backbone
        h = bb.encode(bb.patchify(to_vit_res(img)) + bb.pos_patch, 0, 0)
        hw = (bb.grid, bb.grid)
        for i, blk in enumerate(bb.vit.blocks):
            if i in self.merge_at:
                h, hw = self.merges[self.merge_at.index(i)](h, hw)
            h = blk(h)
        return bb.vit.forward_head(bb.vit.norm(h))
