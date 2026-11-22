"""M1 - coarse look (after Choi 2022): where to look, not what it is.

The 448^2 image is downsampled to 224^2 and passed through the *pretrained* patch-16
embedding and the first `depth` blocks: 196 tokens over a 4x field of view, no new weights
(DESIGN.md fix #3).
"""

import torch.nn as nn

from .backbone import Backbone, to_vit_res


class CoarseLook(nn.Module):
    def __init__(self, dim: int, depth: int, num_classes: int):
        super().__init__()
        self.depth = depth
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, img, bb: Backbone):
        h = bb.encode(bb.patchify(to_vit_res(img)) + bb.pos_patch, 0, self.depth)
        C = h[:, 1:]  # (B, N, D) context map the policy attends over
        g = self.norm(C.mean(1))
        return C, g, self.head(g)
