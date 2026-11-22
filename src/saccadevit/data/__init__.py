from torch.utils.data import DataLoader

from .canvas import Canvas
from .cifar import CIFAR
from .cub import CUB

DATASETS = {"cub": CUB, "cifar100": CIFAR, "canvas": Canvas}


def build_dataset(cfg, split: str, mean, std):
    d = cfg.data
    return DATASETS[d.dataset](d.root, split, d.img_size, mean, std, val_frac=d.val_frac)


def build_loader(cfg, split: str, mean, std, batch_size: int | None = None):
    ds = build_dataset(cfg, split, mean, std)
    train = split == "train"
    return DataLoader(
        ds,
        batch_size=batch_size or cfg.data.batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )
