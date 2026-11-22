import torch

from saccadevit.data.cub import CUB
from saccadevit.models.aggregator import stop_index
from saccadevit.models.glimpse import TopK, action_boxes


def test_cub_geometry_normalised(fake_cub):
    ds = CUB(fake_cub, "train", 448, (0.5,) * 3, (0.5,) * 3, val_frac=0.25)
    s = ds[0]
    assert s["image"].shape == (3, 448, 448) and s["parts"].shape == (15, 3)
    assert 0 <= s["box"].min() and s["box"].max() <= 1


def test_topk_soft_rows_sum_to_one_and_hard_gathers():
    x, scores = torch.randn(2, 196, 8), torch.randn(2, 196, requires_grad=True)
    topk = TopK(sigma=0.05, samples=8).train()
    out, keep = topk(x, scores, 49)
    out.sum().backward()
    assert out.shape == (2, 49, 8) and scores.grad.abs().sum() > 0 and torch.allclose(keep.sum(1), torch.full((2,), 49.0))
    topk.eval()
    idx = scores.topk(49, dim=1).indices.sort(dim=1).values
    assert torch.equal(topk(x, scores, 49)[0], x[torch.arange(2)[:, None], idx])


def test_boxes_stay_inside_image():
    box = action_boxes(torch.tensor([0, 195, 97]), torch.tensor([0.5, 0.5, 1.0]), 14)
    assert box.min() >= 0 and box.max() <= 1


def test_stop_index_first_confident_look():
    ent = torch.tensor([[0.9, 0.1], [0.5, 0.0], [0.2, 0.0]])  # (T+1, N)
    assert stop_index(ent, 0.3).tolist() == [2, 0]
    assert stop_index(ent, 0.0).tolist() == [2, 2]
