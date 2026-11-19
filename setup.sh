#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
assets="$root/assets"
upstream="$assets/H-JEPA"
upstream_commit=97054ecaff83e13839195ec4b92d74c08a0127c5
hf_revision=7b8ba64389690345045650c51c6d6fea0ba4bd02
export HF_HOME="$assets/cache"
export PIP_CACHE_DIR="$assets/cache/pip"

if (( $# != 0 )); then
    printf '%s\n' 'Usage: bash setup.sh' >&2
    exit 2
fi
if ! command -v python3.11 >/dev/null; then
    printf '%s\n' 'Python 3.11 with venv support is required.' >&2
    exit 1
fi

mkdir -p "$assets" "$HF_HOME"
if [[ ! -d "$upstream/.git" ]]; then
    if [[ -e "$upstream" ]]; then
        printf 'Existing source path is not the setup checkout: %s\n' "$upstream" >&2
        exit 1
    fi
    git init -q "$upstream"
fi
if ! git -C "$upstream" cat-file -e "$upstream_commit^{commit}" 2>/dev/null; then
    git -C "$upstream" fetch --depth 1 https://github.com/kevinghst/H-JEPA.git "$upstream_commit"
fi
git -C "$upstream" checkout --detach "$upstream_commit"
if [[ "$(git -C "$upstream" rev-parse HEAD)" != "$upstream_commit" ]] ||
   [[ -n "$(git -C "$upstream" status --porcelain --untracked-files=no)" ]]; then
    printf '%s\n' 'The official source checkout does not match its clean pinned commit.' >&2
    exit 1
fi

python3.11 -m venv "$root/.venv"
python="$root/.venv/bin/python"
"$python" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128'
"$python" -m pip install \
    'transformers==5.9.0' 'huggingface-hub==1.5.0' 'gymnasium==1.2.3' \
    einops numpy scipy h5py hdf5plugin hydra-core loguru rich tqdm
"$python" -m pip check

PYTHONPATH="$upstream/h_jepa:$upstream" PYTHONDONTWRITEBYTECODE=1 "$python" - \
    "$assets" "$upstream_commit" "$hf_revision" <<'PY'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import hdf5plugin
import stable_worldmodel
import torch
from huggingface_hub import snapshot_download
from models.hjepa import HJEPA
from stable_worldmodel.envs.four_room import FourRoomDistractorsEnv
from transformers import ViTModel

assets, upstream_commit, hf_revision = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
repo_id = "jepa-world-models/h-jepa"
paths = {
    str(seed): f"fourroom/fourroom_hjepa_l2/seed{seed}/fourroom_hjepa_l2_object.ckpt"
    for seed in (42, 43, 44)
}
if sys.version_info[:2] != (3, 11) or torch.version.cuda != "12.8":
    raise RuntimeError("Expected Python 3.11 and a PyTorch CUDA 12.8 build")
snapshot_download(
    repo_id=repo_id,
    revision=hf_revision,
    local_dir=assets / "weights",
    allow_patterns=list(paths.values()),
)
weights_sha256 = {}
for seed, relative_path in paths.items():
    path = assets / "weights" / relative_path
    if path.stat().st_size < 1_000_000:
        raise RuntimeError(f"Incomplete checkpoint: {path}")
    with path.open("rb") as f:
        weights_sha256[seed] = hashlib.file_digest(f, "sha256").hexdigest()
receipt = {
    "upstream_commit": upstream_commit,
    "hf_repo": repo_id,
    "hf_revision": hf_revision,
    "weights_sha256": weights_sha256,
    "python": sys.version.split()[0],
    "cuda": torch.version.cuda,
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "transformers", "huggingface-hub", "gymnasium",
                     "einops", "numpy", "scipy", "h5py", "hdf5plugin", "hydra-core",
                     "loguru", "rich", "tqdm")
    },
}
temporary = assets / "setup.json.tmp"
temporary.write_text(json.dumps(receipt, indent=2) + "\n")
temporary.replace(assets / "setup.json")
print(f"Setup complete. Verified inputs recorded in {assets / 'setup.json'}.")
PY
