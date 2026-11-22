"""Fetch and verify everything in configs/assets.yaml.

fetch: datasets -> input/datasets, weights -> input/weights, sources -> third_party/<name>.
Each fetched file's SHA-256 is recorded in input/MANIFEST.json; verify re-checks the
expected file counts and those hashes.
"""

import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from omegaconf import OmegaConf

SPEC = Path("configs/assets.yaml")
MANIFEST = Path("input/MANIFEST.json")
STRIP = {".git", "__pycache__"}
STRIP_SUFFIX = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".pth", ".pt", ".ckpt", ".npz", ".ipynb"}


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def _record(key: str, path: Path) -> None:
    m = _manifest()
    m[key] = {"path": str(path.resolve().relative_to(Path.cwd().resolve())), "sha256": sha256(path)}
    MANIFEST.write_text(json.dumps(m, indent=2))


def fetch_dataset(name: str, d) -> None:
    dest = Path(d.dest)
    dest.mkdir(parents=True, exist_ok=True)
    if _count(d.check.dir) >= d.check.files:
        print(f"[dataset] {name}: present")
        return
    if "torchvision" in d:
        import torchvision.datasets as tvd

        getattr(tvd, d.torchvision)(str(dest), train=True, download=True)
        archive = next(dest.glob("*.tar.gz"))
    else:
        archive = dest / Path(d.url).name
        print(f"[dataset] {name}: downloading {d.url}")
        req = urllib.request.Request(d.url, headers={"User-Agent": "saccadevit-setup/0.1"})  # default Python-urllib gets 403
        with urllib.request.urlopen(req) as r, archive.open("wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        with tarfile.open(archive) as tar:
            tar.extractall(dest, filter="data")
    _record(f"dataset/{name}", archive)
    print(f"[dataset] {name}: {_count(d.check.dir)} files in {d.check.dir}")


def fetch_weights(name: str, w) -> None:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(w.hf_repo, w.file, local_dir=w.dest))
    _record(f"weights/{name}", path)
    print(f"[weights] {name}: {path}")


def fetch_source(name: str, s) -> None:
    dest = Path("third_party") / name
    with tempfile.TemporaryDirectory() as tmp:
        git = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True)  # noqa: E731
        git("init", "-q")
        git("remote", "add", "origin", s.repo)
        if "subdir" in s:
            git("sparse-checkout", "init", "--cone")  # explicit: older git defaults to non-cone
            git("sparse-checkout", "set", s.subdir)
        git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", s.commit)
        git("checkout", "-q", "FETCH_HEAD")
        src = Path(tmp) / s.get("subdir", "")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest, ignore=lambda d, names: [n for n in names if n in STRIP or Path(n).suffix.lower() in STRIP_SUFFIX])
        if "subdir" in s:  # cone mode keeps root files: carry the monorepo licence along
            shutil.copy(Path(tmp) / "LICENSE", dest / "LICENSE")
    (dest / "SNAPSHOT").write_text(f"repo: {s.repo}\ncommit: {s.commit}\nsubdir: {s.get('subdir', '.')}\nlicense: {s.license}\n")
    print(f"[third_party] {name}: {s.repo}@{s.commit[:10]} -> {dest}")


def _count(directory: str) -> int:
    p = Path(directory)
    return sum(1 for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def fetch(only: list[str] | None = None) -> None:
    spec = OmegaConf.load(SPEC)
    Path("input").mkdir(exist_ok=True)
    groups = {"datasets": fetch_dataset, "weights": fetch_weights, "third_party": fetch_source}
    for group, fn in groups.items():
        for name, item in spec[group].items():
            if not only or name in only or group in only:
                fn(name, item)


def verify() -> bool:
    spec = OmegaConf.load(SPEC)
    ok = True
    for name, d in spec.datasets.items():
        n = _count(d.check.dir)
        good = n >= d.check.files
        ok &= good
        print(f"{'ok ' if good else 'BAD'} dataset/{name}: {n}/{d.check.files} files")
    for key, rec in _manifest().items():
        p = Path(rec["path"])
        good = p.exists() and sha256(p) == rec["sha256"]
        ok &= good
        print(f"{'ok ' if good else 'BAD'} {key}: sha256 {rec['sha256'][:12]}")
    for name, s in spec.third_party.items():
        snap = Path("third_party") / name / "SNAPSHOT"
        good = snap.exists() and s.commit in snap.read_text()
        ok &= good
        print(f"{'ok ' if good else 'BAD'} third_party/{name}: {s.commit[:10]}")
    print("all inputs verified" if ok else "missing or changed inputs: run `make setup`")
    return ok
