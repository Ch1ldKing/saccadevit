"""One timm ViT shared by the coarse pass and every glimpse, plus the two full-image baselines."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

VIT_RES = 224


def to_vit_res(img: torch.Tensor) -> torch.Tensor:
    if img.shape[-1] == VIT_RES:
        return img
    return F.interpolate(img, size=VIT_RES, mode="bilinear", antialias=True, align_corners=False)


class Backbone(nn.Module):
    """Exposes patchify / encode so callers control which tokens enter the blocks."""

    def __init__(self, name: str, pretrained: bool, num_classes: int):
        super().__init__()
        self.vit = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        self.dim = self.vit.embed_dim
        self.grid = self.vit.patch_embed.grid_size[0]
        self.num_patches = self.grid**2
        self.depth = len(self.vit.blocks)

    @property
    def norm_stats(self):
        cfg = self.vit.pretrained_cfg
        return tuple(cfg["mean"]), tuple(cfg["std"])

    @property
    def pos_patch(self) -> torch.Tensor:  # (1, N, D)
        return self.vit.pos_embed[:, -self.num_patches:]

    def patchify(self, img224: torch.Tensor) -> torch.Tensor:  # (B, N, D), no position
        return self.vit.patch_embed(img224)

    def encode(self, x: torch.Tensor, start: int = 0, end: int | None = None) -> torch.Tensor:
        """x: patch tokens with position already added. Prepends CLS, runs blocks[start:end]."""
        cls = self.vit.cls_token
        if not self.vit.no_embed_class:
            cls = cls + self.vit.pos_embed[:, :1]
        h = self.vit.norm_pre(torch.cat([cls.expand(len(x), -1, -1), x], 1))
        for blk in self.vit.blocks[start:end]:
            h = blk(h)
        return h


class FullViT(nn.Module):
    """B0: all 196 tokens of the image resized to 224^2."""

    def __init__(self, m):
        super().__init__()
        self.backbone = Backbone(m.backbone, m.pretrained, m.num_classes)

    def forward(self, img):
        return self.backbone.vit(to_vit_res(img))
