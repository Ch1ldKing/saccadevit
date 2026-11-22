"""Synthetic search canvas: one small CIFAR-100 object on a large cluttered canvas.

Objects are 1/16-7/64 of the canvas side (32-56 px at 512^2): CIFAR is natively 32 px, so a
larger canvas only upsamples the same pixels. The coarse 224^2 view sees the target at ~14-24 px.

Clutter = block-shuffled crops of other CIFAR images (natural statistics, no class identity)
over low-frequency noise, so classification needs a look. Val/test canvases are deterministic
per index; train canvases are fresh every epoch.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .cifar import NO_PARTS, cifar_split


def _shuffle_blocks(img: torch.Tensor, block: int, rng) -> torch.Tensor:
    c, h, w = img.shape
    tiles = img.unfold(1, block, block).unfold(2, block, block).reshape(c, -1, block, block)
    tiles = tiles[:, torch.from_numpy(rng.permutation(tiles.shape[1]))]
    n = h // block
    return tiles.reshape(c, n, n, block, block).permute(0, 1, 3, 2, 4).reshape(c, h, w)


class Canvas(Dataset):
    def __init__(self, root, split: str, size: int, mean, std, val_frac: float = 0.1, distractors: int = 8):
        self.images, self.labels = cifar_split(root, split, val_frac)
        self.size, self.target_px, self.distractors = size, (size // 16, size * 7 // 64), distractors
        self.train = split == "train"
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def _patch(self, i: int, px: int) -> torch.Tensor:
        img = torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255
        return F.interpolate(img[None], size=px, mode="bilinear", align_corners=False)[0]

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.train else i)
        S = self.size
        noise = torch.from_numpy(rng.random((3, 8, 8))).float()
        canvas = F.interpolate(noise[None], size=S, mode="bicubic", align_corners=False)[0].clamp(0, 1) * 0.6 + 0.2
        for _ in range(self.distractors):
            px = int(rng.integers(*self.target_px)) // 8 * 8
            j = int(rng.integers(len(self.labels)))
            d = _shuffle_blocks(self._patch(j, px), px // 4, rng)
            x, y = rng.integers(0, S - px, size=2)
            canvas[:, y:y + px, x:x + px] = d
        px = int(rng.integers(*self.target_px))
        x, y = (int(v) for v in rng.integers(0, S - px, size=2))
        canvas[:, y:y + px, x:x + px] = self._patch(i, px)
        if self.train and rng.random() < 0.5:
            canvas, x = canvas.flip(2), S - x - px
        box = torch.tensor([x, y, x + px, y + px], dtype=torch.float) / S
        return {"image": (canvas - self.mean) / self.std, "label": int(self.labels[i]), "box": box, "parts": NO_PARTS}
