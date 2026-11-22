"""YAML configs with single inheritance via a `_base_` key (path relative to the file)."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

EXP_ROOT = Path("configs/exp")


def _load(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    base = cfg.pop("_base_", None)
    return OmegaConf.merge(_load(path.parent / base), cfg) if base else cfg


def run_name(path: Path) -> str:
    """configs/exp/canvas/b0_full.yaml -> canvas-b0_full"""
    path = path.resolve().with_suffix("")
    root = EXP_ROOT.resolve()
    rel = path.relative_to(root) if path.is_relative_to(root) else Path(path.name)
    return "-".join(rel.parts)


def load_config(path: str | Path, overrides: list[str] = ()) -> DictConfig:
    path = Path(path)
    cfg = _load(path)
    cfg.name = run_name(path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg
