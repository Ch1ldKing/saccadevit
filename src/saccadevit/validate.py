"""Load every experiment config, build its model (no weights), and check that each
`train.init` / `eval.ckpt` points at the output of another registered experiment."""

import re
from pathlib import Path

from .config import EXP_ROOT, load_config
from .models import build_model


def validate_all() -> None:
    paths = sorted(EXP_ROOT.rglob("*.yaml"))
    cfgs = {p: load_config(p, ["model.pretrained=false"]) for p in paths}
    names = {c.name for c in cfgs.values()}
    built = {}
    for p, c in cfgs.items():
        key = str(c.model)
        if key not in built:
            built[key] = sum(t.numel() for t in build_model(c.model).parameters()) / 1e6
        refs = [r for r in (c.train.init, c.eval.ckpt) if r and not r.startswith(c.out_dir)]
        for r in refs:
            m = re.match(r"output/([^/]+)/seed\d+/best\.pt$", r)
            assert m and m.group(1) in names, f"{p}: {r} is not produced by any experiment"
        print(f"ok {c.name:28s} {c.model.kind:8s} {built[key]:.2f}M  <- {', '.join(Path(r).parts[1] for r in refs) or '-'}")
    print(f"{len(paths)} configs valid")
