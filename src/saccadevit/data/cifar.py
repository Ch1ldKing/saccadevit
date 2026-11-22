"""CIFAR-100 upsampled to `size`. Control only (appendix B4): no 'where' worth choosing."""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

from .transforms import BoxTransform

NO_PARTS = torch.zeros(15, 3)


def cifar_split(root, split: str, val_frac: float):
    base = CIFAR100(root, train=split != "test", download=False)  # fetched by `make setup`
    if split == "test":
        return base.data, np.array(base.targets)
    perm = np.random.default_rng(0).permutation(len(base.data))
    n_val = int(round(val_frac * len(perm)))
    rows = np.sort(perm[:n_val] if split == "val" else perm[n_val:])
    return base.data[rows], np.array(base.targets)[rows]


class CIFAR(Dataset):
    def __init__(self, root, split: str, size: int, mean, std, val_frac: float = 0.1):
        self.images, self.labels = cifar_split(root, split, val_frac)
        self.tf = BoxTransform(size, mean, std, train=split == "train", scale=(0.8, 1.0))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        x, box, parts = self.tf(img, torch.tensor([0.0, 0.0, 32.0, 32.0]), NO_PARTS)
        return {"image": x, "label": int(self.labels[i]), "box": box, "parts": parts}
