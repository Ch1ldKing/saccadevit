import torch

from .backbone import FullViT
from .merging import MergeViT
from .saccade import SaccadeViT

MODELS = {"full": FullViT, "merge": MergeViT, "saccade": SaccadeViT}


def build_model(m):
    return MODELS[m.kind](m)


def load_weights(model, path) -> None:
    """Non-strict load: later stages start from earlier checkpoints that lack their new modules."""
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    own = model.state_dict()
    state = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
    missing = sorted({k.split(".")[0] for k in own if k not in state})
    model.load_state_dict(state, strict=False)
    print(f"loaded {len(state)}/{len(own)} tensors from {path}; fresh modules: {missing}")
