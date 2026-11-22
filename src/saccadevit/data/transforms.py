"""Joint image + box + keypoint transforms. Geometry is returned normalised to [0, 1]."""

import math
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class BoxTransform:
    """Square output of side `size`. Train: random resized crop + h-flip. Eval: plain resize."""

    def __init__(self, size: int, mean, std, train: bool, scale=(0.6, 1.0)):
        self.size, self.mean, self.std, self.train, self.scale = size, mean, std, train, scale

    def __call__(self, img: Image.Image, box: torch.Tensor, pts: torch.Tensor):
        """box: (4,) xyxy px; pts: (P, 3) x, y px, visible."""
        W, H = img.size
        x0, y0, w, h = self._window(W, H)
        img = TF.resized_crop(img, y0, x0, h, w, [self.size, self.size], antialias=True)
        box = ((box - torch.tensor([x0, y0, x0, y0])) / torch.tensor([w, h, w, h])).clamp(0, 1)
        pts = pts.clone()
        pts[:, :2] = (pts[:, :2] - torch.tensor([x0, y0])) / torch.tensor([w, h])
        inside = (pts[:, :2] >= 0).all(1) & (pts[:, :2] <= 1).all(1)
        pts[:, 2] *= inside.float()
        if self.train and random.random() < 0.5:
            img = TF.hflip(img)
            box = torch.stack([1 - box[2], box[1], 1 - box[0], box[3]])
            pts[:, 0] = 1 - pts[:, 0]
        x = TF.normalize(TF.to_tensor(img), self.mean, self.std)
        return x, box, pts

    def _window(self, W, H):
        if not self.train:
            return 0, 0, W, H
        area = W * H * random.uniform(*self.scale)
        ratio = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
        w = min(W, round(math.sqrt(area * ratio)))
        h = min(H, round(math.sqrt(area / ratio)))
        return random.randint(0, W - w), random.randint(0, H - h), w, h
