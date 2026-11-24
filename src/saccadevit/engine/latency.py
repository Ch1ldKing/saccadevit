"""K2 / H3 / T1-T3: does a FLOP saving become a millisecond saving?

Batch 1 (or eval.latency_batch), real early stopping when tau > 0.
  * eval.graph=true replays CUDA graphs (F1): the whole forward for full-image models and fixed-K
    rollouts; one graph for the coarse pass + one per look when tau > 0. Applied to every model alike.
  * Pass 1: end-to-end timing -> p50 / p95 / peak memory.  Pass 2 (eager only): a sync around each
    section -> per-module breakdown.  FLOPs per image from FlopCounterMode (eager).
Output: latency_<device>[_graph][_b<batch>]_rho<rho>_k<K>.json in the run directory.
"""

import json
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from ..data import build_dataset
from ..models.aggregator import norm_entropy
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


def graph(fn, *inputs):
    """Capture fn(*inputs) as a CUDA graph. Returns run(*new_inputs) -> static outputs."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(*inputs)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(*inputs)

    def run(*xs):
        for buf, x in zip(inputs, xs):
            if buf.data_ptr() != x.data_ptr():
                buf.copy_(x)
        g.replay()
        return out

    return run


class _Autocast:
    """Runs every model entry point under bf16 autocast (weights stay fp32, as in training and eval)."""

    def __init__(self, model):
        self.model, self.m = model, getattr(model, "m", None)

    def __getattr__(self, name):
        fn = getattr(self.model, name)
        return lambda *a, **kw: self._amp(fn, *a, **kw)

    def __call__(self, *a, **kw):
        return self._amp(self.model, *a, **kw)

    @staticmethod
    def _amp(fn, *a, **kw):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return fn(*a, **kw)


def make_forward(model, cfg, example, use_graph: bool):
    """-> f(img) returning looks taken (0 for full-image models). eval.half = bf16 autocast."""
    tau, rho, k = cfg.eval.tau, cfg.eval.keep_ratio, cfg.model.k_max
    if cfg.eval.half and example.is_cuda:
        model = _Autocast(model)
    saccade = isinstance(getattr(model, "model", model), SaccadeViT)
    batched = len(example) > 1
    if not use_graph:
        if not saccade:
            return lambda img: (model(img), 0)[1]
        if batched:
            return lambda img: (model(img, k=k, keep_ratio=rho), k)[1]
        return lambda img: model.infer(img, tau, keep_ratio=rho)[1]
    if not saccade:
        run = graph(model, example.clone())
        return lambda img: (run(img), 0)[1]
    if tau == 0 or batched or model.m.parallel:
        run = graph(lambda x: model.g_all(x, k, rho), example.clone())
        return lambda img: (run(img), k)[1]
    begin = graph(model.g_begin, example.clone())
    C, g, q, e, vis, p0 = begin(example)
    step = graph(lambda img, C, g, q, e, v: model.g_step(img, C, g, q, e, v, rho), example.clone(), C, g, q, e.clone(), vis.clone())

    def f(img):
        C, g, q, e, vis, logits = begin(img)
        for t in range(k):
            if norm_entropy(logits).item() < tau:
                return t
            e, vis, logits = step(img, C, g, q, e, vis)
        return k

    return f


@torch.no_grad()
def run(cfg) -> dict:
    dev = torch.device(cfg.eval.device if cfg.eval.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_graph = bool(cfg.eval.graph) and dev.type == "cuda"
    bs = cfg.eval.latency_batch
    out = out_dir(cfg)
    model = load_for_eval(cfg, dev)
    ds = build_dataset(cfg, "test", *model.backbone.norm_stats)
    dtype = torch.float32
    n = cfg.eval.latency_images
    images = [torch.stack([ds[(i * bs + j) % len(ds)]["image"] for j in range(bs)]).to(dev) for i in range(n)]
    images = [x if x.dtype == torch.uint8 else x.to(dtype) for x in images]
    fwd = make_forward(model, cfg, images[0], use_graph)

    for i in range(cfg.eval.warmup):
        fwd(images[i % n])
    timer, looks, total = Timer(dev), [], []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(cfg.eval.iters):
        img = images[i % n]
        timer.sync()
        t0 = time.perf_counter()
        looks.append(fwd(img))
        timer.sync()
        total.append((time.perf_counter() - t0) * 1e3)
    sections = {}
    if not use_graph and bs == 1 and isinstance(model, SaccadeViT):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(cfg.eval.half) and dev.type == "cuda"):
            for img in images:
                model.infer(img, cfg.eval.tau, keep_ratio=cfg.eval.keep_ratio, timer=timer)
        sections = {k: float(np.sum(v) / n) for k, v in timer.ms.items()}
    eager = make_forward(model, cfg, images[0], False)
    with FlopCounterMode(display=False) as fc:
        for img in images[:16]:
            eager(img)

    p50 = float(np.percentile(total, 50))
    res = {
        "device": str(torch.cuda.get_device_name() if dev.type == "cuda" else "cpu"),
        "graph": use_graph,
        "batch": bs,
        "p50_ms": p50,
        "p95_ms": float(np.percentile(total, 95)),
        "img_per_s": bs * 1e3 / p50,
        "mean_looks": float(np.mean(looks)),
        "gflops_per_img": fc.get_total_flops() / (min(16, n) * bs) / 1e9,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None,
        "sections_ms_per_img": sections,
        "tau": cfg.eval.tau,
        "keep_ratio": cfg.eval.keep_ratio,
        "k_max": cfg.model.k_max if isinstance(model, SaccadeViT) else 0,
    }
    tag = f"{dev.type}{'_graph' if use_graph else ''}{f'_b{bs}' if bs > 1 else ''}"
    if isinstance(model, SaccadeViT):
        tag += f"_rho{cfg.eval.keep_ratio}_k{cfg.model.k_max}" + (f"_tau{cfg.eval.tau}" if cfg.eval.tau else "")
    (out / f"latency_{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res
