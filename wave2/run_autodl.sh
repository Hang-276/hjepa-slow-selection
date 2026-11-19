#!/usr/bin/env bash
# End-to-end Wave-2 orchestrator for an AutoDL vGPU-32GB (single GPU 0).
#
#   bash wave2/run_autodl.sh
#
# Data collection runs on CPU in the background while S2 and S3 run sequentially
# on GPU 0. S1 waits for the pre-registered fourroom_v2 splits, then runs on GPU 0.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

bash wave2/setup_autodl.sh
python="$root/.venv/bin/python"
mkdir -p results

"$python" -B wave2/prepare_data_autodl.py --stage formal \
    > results/wave2_prep.log 2>&1 &
prep=$!
cleanup() {
    if kill -0 "$prep" 2>/dev/null; then
        kill "$prep" 2>/dev/null || true
    fi
}
trap cleanup EXIT

printf '%s\n' 'Running S2 (sprites factorial) on GPU 0 ...'
"$python" -B wave2/s2_sprites_factorial.py --device cuda
printf '%s\n' 'Running S3 (regime switch) on GPU 0 ...'
"$python" -B wave2/s3_sprites_regime_switch.py --device cuda

if ! wait "$prep"; then
    printf '%s\n' 'fourroom_v2 preparation failed; see results/wave2_prep.log' >&2
    exit 1
fi
trap - EXIT

"$python" -B - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
data = root / "assets/data/fourroom_v2"
receipt = json.loads((data / "data_receipt.json").read_text())
expected = "a3f5d768564f0d49f8f99217bfc59ac30d0bff7672d4ba4c7093f05b708fbdb3"
actual = hashlib.sha256((data / "splits_v2.json").read_bytes()).hexdigest()
if actual != expected or receipt["splits_sha256"] != expected:
    raise SystemExit(f"fourroom_v2 split hash mismatch: {actual}")
print("fourroom_v2 splits verified:", actual)
PY

printf '%s\n' 'Running S1 (FourRoom 192-D projections) on GPU 0 ...'
"$python" -B wave2/s1_fourroom_dim192.py --device cuda

printf '%s\n' 'Verifying Wave-2 and building tarballs ...'
"$python" -B wave2/verify_wave2.py
