#!/usr/bin/env bash
# ResPlan (https://github.com/m-agour/ResPlan) into SOURCE (./source by default): ResPlan.pkl, which
# compare_resplan.py compares the synthetic plans with.
set -euo pipefail

SOURCE="${1:-source}"
mkdir -p "$SOURCE"
cd "$SOURCE"
[ -f ResPlan.zip ] || curl -L -o ResPlan.zip https://github.com/m-agour/ResPlan/raw/main/ResPlan.zip
[ -f ResPlan.pkl ] || unzip -o ResPlan.zip
