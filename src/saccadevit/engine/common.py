import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from ..models import build_model, load_weights


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast(cfg, dev: torch.device):
    return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cfg.train.amp and dev.type == "cuda")


def to_device(batch: dict, dev) -> dict:
    return {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_for_eval(cfg, dev):
    model = build_model(cfg.model)
    load_weights(model, cfg.eval.ckpt)
    return model.to(dev).eval()


def out_dir(cfg) -> Path:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml")
    return out
