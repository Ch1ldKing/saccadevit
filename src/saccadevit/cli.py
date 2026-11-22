"""sv train|eval|sweep|latency CONFIG [key=value ...]; sv compare A B; sv assets fetch|verify; sv validate"""

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
    c.add_argument("--rule", default="last", help="'last' | 'k=2' | 'tau=0.4', for both sides")
    c.add_argument("--rule-a", help="override for A")
    c.add_argument("--rule-b", help="override for B")
    c.add_argument("--name", help="save to output/compare/<name>.json (picked up by `sv report`)")
    sub.add_parser("report", help="collect output/ into docs/CURRENT_RESULTS.md")
    a = sub.add_parser("assets", help="fetch / verify input/ and third_party/ (configs/assets.yaml)")
    a.add_argument("action", choices=["fetch", "verify"])
    a.add_argument("--only", nargs="*", help="asset names or groups (datasets, weights, third_party)")
    sub.add_parser("validate", help="load every experiment config and build its model")
    args = p.parse_args()

    if args.cmd == "assets":
        from . import assets

        if args.action == "fetch":
            assets.fetch(args.only)
        else:
            raise SystemExit(0 if assets.verify() else 1)
        return
    if args.cmd == "report":
        from .engine.report import write

        write()
        return
    if args.cmd == "validate":
        from .validate import validate_all

        validate_all()
        return
    if args.cmd == "compare":
        from .engine.stats import compare

        compare(args.a, args.b, args.rule_a or args.rule, args.rule_b or args.rule, args.name)
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
