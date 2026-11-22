"""K2 / H3: does a FLOP saving become a millisecond saving? Batch 1, real early stopping.

Two passes: (1) end-to-end timing without inner syncs -> p50 / p95 / peak memory;
(2) a profiling pass with a sync around each section -> per-module breakdown.
FLOPs per image come from torch's FlopCounterMode.
"""

import json
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from ..data import build_dataset
from ..models.saccade import SaccadeViT
from .common import load_for_eval, out_dir


class Timer:
    def __init__(self, dev):
        self.dev, self.ms = dev, defaultdict(list)

    def sync(self):
        if self.dev.type == "cuda":
            torch.cuda.synchronize()

    @contextmanager
    def section(self, name):
        self.sync()
        t0 = time.perf_counter()
        yield
        self.sync()
        self.ms[name].append((time.perf_counter() - t0) * 1e3)


def _forward(model, img, cfg, timer=None):
    if isinstance(model, SaccadeViT):
        return model.infer(img, cfg.eval.tau, keep_ratio=cfg.eval.keep_ratio, timer=timer)[1]
    model(img)
    return 0


@torch.no_grad()
def run(cfg) -> dict:
    dev = torch.device(cfg.eval.device if cfg.eval.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    out = out_dir(cfg)
    model = load_for_eval(cfg, dev)
    if cfg.eval.half:
        model = model.to(torch.bfloat16)
    ds = build_dataset(cfg, "test", *model.backbone.norm_stats)
    dtype = torch.bfloat16 if cfg.eval.half else torch.float32
    images = [ds[i]["image"][None].to(dev, dtype) for i in range(cfg.eval.latency_images)]

    for i in range(cfg.eval.warmup):
        _forward(model, images[i % len(images)], cfg)
    timer, looks, total = Timer(dev), [], []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(cfg.eval.iters):
        img = images[i % len(images)]
        timer.sync()
        t0 = time.perf_counter()
        looks.append(_forward(model, img, cfg))
        timer.sync()
        total.append((time.perf_counter() - t0) * 1e3)
    for img in images:
        _forward(model, img, cfg, timer)
    with FlopCounterMode(display=False) as fc:
        for img in images[:16]:
            _forward(model, img, cfg)

    res = {
        "device": str(torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"),
        "p50_ms": float(np.percentile(total, 50)),
        "p95_ms": float(np.percentile(total, 95)),
        "mean_looks": float(np.mean(looks)),
        "gflops_per_img": fc.get_total_flops() / min(16, len(images)) / 1e9,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None,
        "sections_ms_per_img": {k: float(np.sum(v) / len(images)) for k, v in timer.ms.items()},
        "tau": cfg.eval.tau,
        "keep_ratio": cfg.eval.keep_ratio,
    }
    (out / f"latency_{dev.type}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res
