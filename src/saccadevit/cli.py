"""sv train|eval|sweep|latency CONFIG [key=value ...];  sv compare GLOB_A GLOB_B [--k K | --tau T]"""

import argparse

from .config import load_config


def main() -> None:
    p = argparse.ArgumentParser(prog="sv")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval", "sweep", "latency"):
        s = sub.add_parser(name)
        s.add_argument("config")
        s.add_argument("overrides", nargs="*", help="OmegaConf dotlist, e.g. seed=1 train.epochs=5")
        if name == "eval":
            s.add_argument("--split", default="test")
    c = sub.add_parser("compare", help="paired bootstrap B - A over eval_*.npz files (one per seed)")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--k", type=int)
    c.add_argument("--tau", type=float)
    args = p.parse_args()

    if args.cmd == "compare":
        from .engine.stats import compare

        compare(args.a, args.b, args.k, args.tau)
        return
    cfg = load_config(args.config, args.overrides)
    if args.cmd == "train":
        from .engine.train import run

        run(cfg)
    elif args.cmd == "eval":
        from .engine.evaluate import run

        run(cfg, args.split)
    elif args.cmd == "sweep":
        from .engine.sweep import run

        run(cfg)
    else:
        from .engine.latency import run

        run(cfg)
