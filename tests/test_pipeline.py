"""End-to-end on fake CUB: every stage trains one epoch, evaluation and latency run."""

import json

import pytest
import torch

from saccadevit.engine import evaluate, latency, sweep, train
from saccadevit.models import build_model


@pytest.mark.parametrize("policy", ["learned", "random", "saliency", "center", "gt", "mix"])
def test_every_policy_rolls_out(cfg, policy):
    c = cfg("b5_ours")
    model = build_model(c.model)
    img = torch.randn(2, 3, 448, 448)
    box, parts = torch.tensor([[0.1, 0.1, 0.5, 0.6]] * 2), torch.rand(2, 15, 3)
    r = model(img, policy=policy, greedy=False, box=box, parts=parts)
    assert r.logits.shape == (c.model.k_max + 1, 2, 200)
    assert (r.logp is not None) == (policy == "learned")


def test_stages_chain(cfg, tmp_path):
    b0 = cfg("b0_full")
    train.run(b0)
    a = cfg("stage_a", f"train.init={b0.out_dir}/best.pt")
    train.run(a)
    b = cfg("b5_ours", f"train.init={a.out_dir}/best.pt")
    train.run(b)
    summary = evaluate.run(b, "test")
    assert {"acc@0", "acc@3", "tau"} <= summary.keys()
    sweep.run(cfg("a3_keep_sweep", f"eval.ckpt={b.out_dir}/best.pt", "eval.keep_ratios=[0.1,1.0]"))
    lat = latency.run(cfg("b5_ours", f"eval.ckpt={b.out_dir}/best.pt", "eval.latency_images=2",
                          "eval.warmup=1", "eval.iters=3", "eval.tau=0.5"))
    assert lat["p50_ms"] > 0 and set(lat["sections_ms_per_img"]) >= {"coarse"}
    train.run(cfg("b4_merge", f"train.init={b0.out_dir}/best.pt"))
    assert json.loads((tmp_path / "b5_ours" / "summary_test.json").read_text())
