#!/usr/bin/env bash
# Single-GPU AutoDL setup for Wave-2 (vGPU-32GB, GPU 0).
# Reuses an existing CUDA-enabled PyTorch when present; otherwise installs a
# CUDA 12.8 wheel. No secrets are printed.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
assets="$root/assets"
upstream="$assets/H-JEPA"
upstream_commit=97054ecaff83e13839195ec4b92d74c08a0127c5
hf_revision=7b8ba64389690345045650c51c6d6fea0ba4bd02
export HF_HOME="$assets/cache"
export PIP_CACHE_DIR="$assets/cache/pip"

mkdir -p "$assets" "$HF_HOME" "$PIP_CACHE_DIR"

if (( $# != 0 )); then
    printf '%s\n' 'Usage: bash wave2/setup_autodl.sh' >&2
    exit 2
fi

if ! command -v git >/dev/null 2>&1; then
    printf '%s\n' 'git not found; attempting apt-get install git' >&2
    apt-get update -y
    apt-get install -y git
fi

version_ok() {
    "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
        >/dev/null 2>&1
}
PY=""
for candidate in python3.11 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && version_ok "$candidate"; then
        PY="$candidate"
        break
    fi
done
if [[ -z "$PY" ]]; then
    if command -v conda >/dev/null 2>&1; then
        printf '%s\n' 'System Python < 3.11; creating conda env at .conda311' >&2
        conda create -y -p "$root/.conda311" python=3.11
        PY="$root/.conda311/bin/python"
    else
        printf '%s\n' 'Python >= 3.11 (or conda) is required.' >&2
        exit 1
    fi
fi

if [[ ! -x "$root/.venv/bin/python" ]]; then
    if "$PY" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
        printf '%s\n' 'Reusing existing CUDA-enabled PyTorch via --system-site-packages'
        "$PY" -m venv --system-site-packages "$root/.venv"
    else
        "$PY" -m venv "$root/.venv"
    fi
fi
python="$root/.venv/bin/python"
"$python" -m pip install --upgrade pip
if ! "$python" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    printf '%s\n' 'Installing PyTorch CUDA 12.8 wheels'
    "$python" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision
fi
"$python" -m pip install \
    'transformers==5.9.0' 'huggingface-hub==1.5.0' 'gymnasium==1.2.3' \
    einops numpy scipy h5py hdf5plugin hydra-core omegaconf loguru rich tqdm pillow
"$python" -m pip check || printf '%s\n' 'pip check reported environment warnings; continuing' >&2
"$python" -c 'import torch, transformers, h5py, hdf5plugin, hydra, gymnasium, einops, numpy, scipy, PIL; assert torch.cuda.is_available()'

if [[ ! -d "$upstream/.git" ]]; then
    if [[ -e "$upstream" ]]; then
        printf 'Existing source path is not the pinned checkout: %s\n' "$upstream" >&2
        exit 1
    fi
    git init -q "$upstream"
fi
if ! git -C "$upstream" cat-file -e "$upstream_commit^{commit}" 2>/dev/null; then
    git -C "$upstream" fetch --depth 1 https://github.com/kevinghst/H-JEPA.git \
        "$upstream_commit"
fi
git -C "$upstream" checkout --detach "$upstream_commit"
if [[ "$(git -C "$upstream" rev-parse HEAD)" != "$upstream_commit" ]] ||
   [[ -n "$(git -C "$upstream" status --porcelain --untracked-files=no)" ]]; then
    printf '%s\n' 'The official source checkout does not match its clean pinned commit.' >&2
    exit 1
fi

PYTHONPATH="$upstream/h_jepa:$upstream" PYTHONDONTWRITEBYTECODE=1 "$python" - \
    "$assets" "$upstream_commit" "$hf_revision" <<'PY'
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import hdf5plugin  # noqa: F401
import stable_worldmodel  # noqa: F401
import torch
from huggingface_hub import HfApi, snapshot_download
from models.hjepa import HJEPA  # noqa: F401

assets, upstream_commit, hf_revision = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
repo_id = "jepa-world-models/h-jepa"
tags = ("fourroom_lewm", "fourroom_hjepa_l2", "fourroom_hjepa_l4")
seeds = (42, 43, 44)

if sys.version_info[:2] < (3, 11):
    raise RuntimeError("Python >= 3.11 is required")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA-enabled PyTorch is required")

wanted = []
for tag in tags:
    for seed in seeds:
        wanted.append(f"fourroom/{tag}/seed{seed}/{tag}_object.ckpt")
        wanted.append(f"fourroom/{tag}/seed{seed}/normalizer.pt")

api = HfApi()
info = api.model_info(repo_id, revision=hf_revision, files_metadata=True)
if info.sha != hf_revision:
    raise RuntimeError(f"Hub resolved {info.sha!r}, expected {hf_revision!r}")
remote = {s.rfilename: s for s in info.siblings}
missing = [name for name in wanted if name not in remote]
if missing:
    raise RuntimeError(f"Pinned revision is missing: {missing}")

snapshot_download(repo_id=repo_id, revision=hf_revision,
                  local_dir=assets / "weights", allow_patterns=wanted)

def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

checkpoints, normalizers = {}, {}
for tag in tags:
    for seed in seeds:
        ckpt = assets / "weights/fourroom" / tag / f"seed{seed}" / f"{tag}_object.ckpt"
        norm = assets / "weights/fourroom" / tag / f"seed{seed}" / "normalizer.pt"
        for path, bucket, minimum in ((ckpt, checkpoints, 1_000_000),
                                      (norm, normalizers, 1)):
            relative = path.relative_to(assets / "weights").as_posix()
            expected = remote[relative]
            if not path.is_file() or path.stat().st_size < minimum:
                raise RuntimeError(f"Incomplete file: {path}")
            digest = sha256(path)
            if path.stat().st_size != expected.size:
                raise RuntimeError(f"Size mismatch: {path}")
            if expected.lfs is not None and digest != expected.lfs.sha256:
                raise RuntimeError(f"LFS SHA256 mismatch: {path}")
            entry = {"relative_path": relative, "bytes": path.stat().st_size,
                     "sha256": digest}
            if expected.lfs is not None:
                entry["hf_lfs_sha256"] = expected.lfs.sha256
            bucket[f"{tag}/seed{seed}"] = entry

l2_hashes = {str(seed): checkpoints[f"fourroom_hjepa_l2/seed{seed}"]["sha256"]
             for seed in seeds}
packages = {}
for name in ("torch", "torchvision", "transformers", "huggingface-hub",
             "gymnasium", "einops", "numpy", "scipy", "h5py", "hdf5plugin",
             "hydra-core", "omegaconf", "loguru", "rich", "tqdm", "pillow"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None

setup = {
    "upstream_commit": upstream_commit,
    "hf_repo": repo_id,
    "hf_revision": hf_revision,
    "weights_sha256": l2_hashes,
    "python": sys.version.split()[0],
    "cuda": torch.version.cuda,
    "packages": packages,
    "hf_endpoint": os.environ.get("HF_ENDPOINT"),
}
setup_v2 = {
    "hf_repo": repo_id,
    "hf_revision": hf_revision,
    "resolved_hub_sha": info.sha,
    "checkpoints": checkpoints,
    "normalizers": normalizers,
    "note": "Wave-2 AutoDL weights receipt.",
}
for name, payload in (("setup.json", setup), ("setup_v2.json", setup_v2)):
    temporary = assets / f"{name}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    temporary.replace(assets / name)
print(f"Setup complete. Verified {len(checkpoints)} checkpoints and "
      f"{len(normalizers)} normalizers in {assets}.")
PY
