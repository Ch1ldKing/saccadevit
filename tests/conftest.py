import numpy as np
import pytest
from PIL import Image

from saccadevit.config import load_config


@pytest.fixture(scope="session")
def fake_cub(tmp_path_factory):
    """A 12-image CUB_200_2011 lookalike (same files and formats)."""
    root = tmp_path_factory.mktemp("CUB_200_2011")
    (root / "images" / "001.A").mkdir(parents=True)
    (root / "parts").mkdir()
    rng = np.random.default_rng(0)
    imgs, labels, split, boxes, parts = [], [], [], [], []
    for i in range(1, 13):
        name = f"001.A/img_{i}.jpg"
        Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(root / "images" / name)
        imgs.append(f"{i} {name}")
        labels.append(f"{i} {i % 3 + 1}")
        split.append(f"{i} {int(i <= 8)}")
        boxes.append(f"{i} 20.0 10.0 80.0 60.0")
        parts += [f"{i} {p} {30 + p}.0 {20 + p}.0 {int(p % 2)}" for p in range(1, 16)]
    for fname, rows in [("images.txt", imgs), ("image_class_labels.txt", labels), ("train_test_split.txt", split),
                        ("bounding_boxes.txt", boxes), ("parts/part_locs.txt", parts)]:
        (root / fname).write_text("\n".join(rows) + "\n")
    return root


@pytest.fixture
def cfg(fake_cub, tmp_path):
    def make(exp: str, *overrides):
        return load_config(f"configs/exp/{exp}.yaml", [
            f"data.root={fake_cub}", "data.batch_size=2", "data.num_workers=0", "data.val_frac=0.25",
            "model.pretrained=false", "train.epochs=1", "train.warmup_epochs=0", "eval.batch_size=4",
            f"out_dir={tmp_path}/{exp}", *overrides,
        ])
    return make
