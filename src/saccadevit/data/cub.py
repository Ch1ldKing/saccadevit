"""CUB-200-2011 with bounding boxes and the 15 part locations.

Splits: official test = `test`; official train is split once into `train` / `val`
(val_frac, fixed seed 0) so tau calibration never touches the test set.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import BoxTransform

NUM_PARTS = 15


def _table(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64, ndmin=2)


def split_rows(is_train: np.ndarray, split: str, val_frac: float) -> np.ndarray:
    train = np.flatnonzero(is_train)
    perm = np.random.default_rng(0).permutation(len(train))
    n_val = int(round(val_frac * len(train)))
    rows = {"train": train[perm[n_val:]], "val": train[perm[:n_val]], "test": np.flatnonzero(~is_train)}[split]
    return np.sort(rows)


class CUB(Dataset):
    def __init__(self, root, split: str, size: int, mean, std, val_frac: float = 0.1):
        root = Path(root)
        names = [line.split()[1] for line in (root / "images.txt").read_text().splitlines()]
        labels = _table(root / "image_class_labels.txt")[:, 1].astype(int) - 1
        is_train = _table(root / "train_test_split.txt")[:, 1].astype(bool)
        x, y, w, h = _table(root / "bounding_boxes.txt")[:, 1:].T
        parts = _table(root / "parts" / "part_locs.txt")  # img_id part_id x y visible

        pts = torch.zeros(len(names), NUM_PARTS, 3)
        pts[parts[:, 0].astype(int) - 1, parts[:, 1].astype(int) - 1] = torch.tensor(parts[:, 2:5], dtype=torch.float)

        rows = split_rows(is_train, split, val_frac)
        self.paths = [root / "images" / names[r] for r in rows]
        self.labels = labels[rows].tolist()
        self.boxes = torch.tensor(np.stack([x, y, x + w, y + h], 1)[rows], dtype=torch.float)
        self.parts = pts[rows]
        self.tf = BoxTransform(size, mean, std, train=split == "train")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        x, box, parts = self.tf(img, self.boxes[i], self.parts[i])
        return {"image": x, "label": self.labels[i], "box": box, "parts": parts}
