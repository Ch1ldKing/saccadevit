"""H1: paired bootstrap of accuracy differences over the same test images, pooled across seeds,
reported overall and per stratum (coarse pass right/wrong, object-size tercile)."""

import glob
import json

import numpy as np
import torch

from ..models.aggregator import stop_index


def _correct(path: str, k: int | None, tau: float | None) -> tuple[np.ndarray, dict]:
    r = dict(np.load(path))
    c = r["correct"]
    if tau is not None:
        looks = stop_index(torch.from_numpy(r["entropy"]), tau).numpy()
        vec = c[looks, np.arange(c.shape[1])]
    else:
        vec = c[-1 if k is None else k]
    return vec.astype(float), r


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0) -> dict:
    """a, b: (seeds, images) correctness. Resamples images; returns mean(b - a) with 95% CI."""
    diff = (b - a).mean(0)
    idx = np.random.default_rng(seed).integers(0, len(diff), size=(n, len(diff)))
    boots = diff[idx].mean(1)
    return {"diff": float(diff.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "p_le_0": float((boots <= 0).mean()), "n": int(len(diff))}


def compare(glob_a: str, glob_b: str, k=None, tau=None) -> dict:
    pa, pb = sorted(glob.glob(glob_a)), sorted(glob.glob(glob_b))
    assert len(pa) == len(pb) > 0, (pa, pb)
    A, B, ref = [], [], None
    for x, y in zip(pa, pb):
        ca, ra = _correct(x, k, tau)
        cb, _ = _correct(y, k, tau)
        A.append(ca)
        B.append(cb)
        ref = ra if ref is None else ref
    A, B = np.stack(A), np.stack(B)
    strata = {"all": np.ones(A.shape[1], bool)}
    coarse_ok = ref["correct"][0].astype(bool)
    strata |= {"coarse_right": coarse_ok, "coarse_wrong": ~coarse_ok}
    q1, q2 = np.quantile(ref["area"], [1 / 3, 2 / 3])
    strata |= {"small": ref["area"] <= q1, "medium": (ref["area"] > q1) & (ref["area"] <= q2), "large": ref["area"] > q2}
    res = {"a": pa, "b": pb, "acc_a": float(A.mean()), "acc_b": float(B.mean())}
    res |= {name: paired_bootstrap(A[:, m], B[:, m]) for name, m in strata.items() if m.any()}
    print(json.dumps(res, indent=2))
    return res
