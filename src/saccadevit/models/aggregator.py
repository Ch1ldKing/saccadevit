"""M4 - accumulate and stop (our addition 2): e_t = GRU(e_{t-1}, z_t); stop if H(p) < tau or t = K."""

import math

import torch
import torch.nn as nn


class Aggregator(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.init = nn.Linear(dim, dim)
        self.cell = nn.GRUCell(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def start(self, g):
        return torch.tanh(self.init(g.float()))

    def step(self, e, z):
        e = self.cell(z.float(), e)
        return e, self.head(self.norm(e))


def norm_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Prediction entropy divided by log(C), in [0, 1]."""
    logp = logits.float().log_softmax(-1)
    return -(logp.exp() * logp).sum(-1) / math.log(logits.shape[-1])


def stop_index(entropy: torch.Tensor, tau: float) -> torch.Tensor:
    """entropy: (T+1, N), row 0 = coarse pass. Index of the first row under tau (else T) = looks taken."""
    T = entropy.shape[0] - 1
    below = entropy < tau
    below[-1] = True
    return below.float().argmax(0).clamp(max=T)
