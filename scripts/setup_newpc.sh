#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p output/logs
status_file=output/logs/setup_newpc.status
printf 'running\n' > "$status_file"

finish() {
    local result=$?
    local pid
    for pid in ${cifar_pid:-} ${cub_pid:-}; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    if (( result == 0 )); then
        printf 'complete\n' > "$status_file"
    else
        printf 'failed (%s)\n' "$result" > "$status_file"
    fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export PATH="$HOME/.local/bin:$PATH"
export UV_HTTP_TIMEOUT=120
export UV_NO_PROGRESS=1
export PYTHONUNBUFFERED=1

step() {
    printf '\n[%s] %s\n' "$(date '+%F %T %Z')" "$*"
}

get_archive() {
    local archive=$1 checksum=$2 mirror=$3 upstream=$4
    mkdir -p "$(dirname "$archive")"
    if [[ -f "$archive" ]] && printf '%s  %s\n' "$checksum" "$archive" | md5sum -c --status; then
        return
    fi

    local args=(-fsSL --retry 3 --connect-timeout 20 --speed-time 120 --speed-limit 1024
        --user-agent saccadevit-setup/0.1 --continue-at -)
    if curl "${args[@]}" --output "$archive.mirror.part" "$mirror" \
        && printf '%s  %s\n' "$checksum" "$archive.mirror.part" | md5sum -c; then
        mv "$archive.mirror.part" "$archive"
        return
    fi
    step "Trying the upstream source for $(basename "$archive")"
    curl "${args[@]}" --output "$archive.part" "$upstream"
    printf '%s  %s\n' "$checksum" "$archive.part" | md5sum -c
    mv "$archive.part" "$archive"
}

get_cifar_archive() {
    local root=input/datasets/cifar100
    if [[ -f "$root/cifar-100-python/train" && -f "$root/cifar-100-python/test" && -f "$root/cifar-100-python/meta" ]]; then
        return
    fi
    step 'Downloading CIFAR-100'
    get_archive "$root/cifar-100-python.tar.gz" eb9058c3a382ffc7106e4002c42a8d85 \
        'https://scidata.sjtu.edu.cn/records/xk2s3-v1e12/files/cifar-100-python.tar.gz?download=1' \
        'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
}

get_cub_archive() {
    local root=input/datasets/CUB_200_2011
    if [[ -d "$root/images" && -f "$root/parts/part_locs.txt" ]] \
        && [[ $(find "$root/images" -type f -name '*.jpg' | wc -l) -ge 11788 ]]; then
        return
    fi
    step 'Downloading CUB-200-2011'
    get_archive input/datasets/CUB_200_2011.tgz 97eceeb196236b17998738112f37df78 \
        'https://huggingface.co/datasets/XiN0919/FGVC/resolve/main/CUB_200_2011.tgz' \
        'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
}

step 'Checking the NVIDIA GPU'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
get_cifar_archive &
cifar_pid=$!
get_cub_archive &
cub_pid=$!

step 'Installing Python 3.11 and locked dependencies'
if ! command -v uv >/dev/null 2>&1; then
    curl -fsSL --retry 4 --connect-timeout 20 https://astral.sh/uv/install.sh \
        | env UV_UNMANAGED_INSTALL="$HOME/.local/bin" sh
fi
uv python install 3.11
uv sync --locked --python 3.11

step 'Checking the PyTorch GPU, BF16 and CUDA Graph paths'
uv run --no-sync python - <<'PY'
import platform

import torch
import torchvision

from saccadevit.config import load_config
from saccadevit.engine.latency import make_forward
from saccadevit.models import build_model

print(f"Python {platform.python_version()}, torch {torch.__version__}, torchvision {torchvision.__version__}")
assert torch.cuda.is_available(), "PyTorch could not access the NVIDIA GPU"
assert torch.cuda.is_bf16_supported(including_emulation=False), "GPU BF16 is unavailable"
print(f"CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name()}")
cfg = load_config("configs/exp/b5_par.yaml", ["model.pretrained=false", "model.k_max=2", "eval.half=true"])
model = build_model(cfg.model).cuda().eval()
example = torch.zeros((1, 3, 448, 448), device="cuda", dtype=torch.uint8)
with torch.no_grad():
    forward = make_forward(model, cfg, example, use_graph=True)
    assert forward(example) == cfg.model.k_max
    torch.cuda.synchronize()
print("B5-par CUDA Graph inference with BF16 passed")
PY

step 'Fetching the ViT-Tiny weights'
weight=vit_tiny_patch16_224.augreg_in21k_ft_in1k
if [[ ! -s "input/weights/$weight/model.safetensors" ]]; then
    uv run --no-sync sv assets fetch --only "$weight" \
        || env HF_ENDPOINT=https://hf-mirror.com uv run --no-sync sv assets fetch --only "$weight"
fi

step 'Checking and extracting CIFAR-100'
wait "$cifar_pid"
cifar_pid=
uv run --no-sync sv assets fetch --only cifar100

step 'Checking and extracting CUB-200-2011'
wait "$cub_pid"
cub_pid=
uv run --no-sync python - <<'PY'
import tarfile
from pathlib import Path

from saccadevit.assets import _record

root = Path("input/datasets")
archive = root / "CUB_200_2011.tgz"
images = root / "CUB_200_2011/images"
if sum(1 for path in images.rglob("*.jpg") if path.is_file()) < 11788:
    with tarfile.open(archive) as dataset:
        dataset.extractall(root, filter="data")
if archive.is_file():
    _record("dataset/cub", archive)
PY
uv run --no-sync sv assets verify

step 'Running configuration and pipeline tests'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-sync sv validate
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --no-sync pytest -q
step 'SaccadeViT environment is ready'
