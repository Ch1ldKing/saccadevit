#!/usr/bin/env bash
# CUB-200-2011 (~1.1 GB) into data/. CIFAR-100 is fetched by torchvision on first use.
set -euo pipefail
mkdir -p data
if [ ! -d data/CUB_200_2011 ]; then
  curl -L -o data/CUB_200_2011.tgz https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
  tar -xzf data/CUB_200_2011.tgz -C data
  rm data/CUB_200_2011.tgz
fi
ls data/CUB_200_2011
