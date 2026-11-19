"""Download the Wave-1 extra FourRoom checkpoints (LeWM L=1 and H-JEPA L=4).

Pinned to the same Hugging Face revision used by setup.sh. Every downloaded
file is checked against the Hub's size and LFS SHA256 before an entry is
written to assets/setup_v2.json. assets/setup.json is never modified.
"""

import os

_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if _vis not in {"6", "7", "6,7", "7,6"}:
    raise RuntimeError(
        f"FATAL GPU ISOLATION GUARD: CUDA_VISIBLE_DEVICES={_vis!r}; only GPUs 6 and 7 are allowed!"
    )

import argparse
import hashlib
import json
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
HF_REPO = "jepa-world-models/h-jepa"
HF_REVISION = "7b8ba64389690345045650c51c6d6fea0ba4bd02"
MODEL_TAGS = ("fourroom_lewm", "fourroom_hjepa_l4")
SEEDS = (42, 43, 44)
EXISTING_TAGS = ("fourroom_hjepa_l2",)
MIN_CHECKPOINT_BYTES = 1_000_000


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_path(tag, seed):
    return ASSETS / "weights/fourroom" / tag / f"seed{seed}" / f"{tag}_object.ckpt"


def normalizer_path(tag, seed):
    return ASSETS / "weights/fourroom" / tag / f"seed{seed}" / "normalizer.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List and verify remote metadata only")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ASSETS / "cache"))
    api = HfApi()
    info = api.model_info(HF_REPO, revision=HF_REVISION, files_metadata=True)
    if info.sha != HF_REVISION:
        raise RuntimeError(f"Hub resolved {info.sha!r}, expected pinned {HF_REVISION!r}")
    remote = {sibling.rfilename: sibling for sibling in info.siblings}

    wanted = []
    for tag in MODEL_TAGS:
        for seed in SEEDS:
            wanted.append(f"fourroom/{tag}/seed{seed}/{tag}_object.ckpt")
            wanted.append(f"fourroom/{tag}/seed{seed}/normalizer.pt")
    for tag in EXISTING_TAGS:
        for seed in SEEDS:
            wanted.append(f"fourroom/{tag}/seed{seed}/normalizer.pt")
    missing = [name for name in wanted if name not in remote]
    if missing:
        raise RuntimeError(f"Pinned revision is missing required files: {missing}")

    for name in wanted:
        entry = remote[name]
        print(f"{name}\t{entry.size} bytes\t{entry.lfs.sha256 if entry.lfs else 'no-lfs'}")
    if args.dry_run:
        return

    snapshot_download(
        repo_id=HF_REPO,
        revision=HF_REVISION,
        local_dir=ASSETS / "weights",
        allow_patterns=wanted,
    )

    checkpoints = {}
    normalizers = {}
    for tag in MODEL_TAGS:
        for seed in SEEDS:
            path = checkpoint_path(tag, seed)
            if not path.is_file():
                raise FileNotFoundError(f"Download did not produce {path}")
            if path.stat().st_size < MIN_CHECKPOINT_BYTES:
                raise RuntimeError(f"Checkpoint is suspiciously small: {path}")
            relative = path.relative_to(ASSETS / "weights").as_posix()
            expected = remote[relative]
            actual = sha256(path)
            if path.stat().st_size != expected.size:
                raise RuntimeError(f"Size mismatch for {path}: {path.stat().st_size} != {expected.size}")
            if expected.lfs is not None and actual != expected.lfs.sha256:
                raise RuntimeError(f"LFS SHA256 mismatch for {path}")
            checkpoints[f"{tag}/seed{seed}"] = {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": actual,
                "hf_lfs_sha256": expected.lfs.sha256 if expected.lfs else None,
            }

    for tag in MODEL_TAGS + EXISTING_TAGS:
        for seed in SEEDS:
            path = normalizer_path(tag, seed)
            if not path.is_file():
                raise FileNotFoundError(f"Download did not produce {path}")
            relative = path.relative_to(ASSETS / "weights").as_posix()
            expected = remote[relative]
            actual = sha256(path)
            if path.stat().st_size != expected.size:
                raise RuntimeError(f"Size mismatch for {path}: {path.stat().st_size} != {expected.size}")
            if expected.lfs is not None and actual != expected.lfs.sha256:
                raise RuntimeError(f"LFS SHA256 mismatch for {path}")
            normalizers[f"{tag}/seed{seed}"] = {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": actual,
            }

    setup = json.loads((ASSETS / "setup.json").read_text())
    for seed in SEEDS:
        relative = f"fourroom/fourroom_hjepa_l2/seed{seed}/fourroom_hjepa_l2_object.ckpt"
        path = ASSETS / "weights" / relative
        if sha256(path) != setup["weights_sha256"][str(seed)]:
            raise RuntimeError(f"Existing L=2 checkpoint no longer matches setup.json: {path}")
        expected = remote[relative]
        checkpoints[f"fourroom_hjepa_l2/seed{seed}"] = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": setup["weights_sha256"][str(seed)],
            "hf_lfs_sha256": expected.lfs.sha256 if expected.lfs else None,
        }

    receipt = {
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "resolved_hub_sha": info.sha,
        "written": "2026-11-17",
        "checkpoints": checkpoints,
        "normalizers": normalizers,
        "note": "Wave-1 auxiliary download. The original assets/setup.json is unchanged.",
    }
    temporary = ASSETS / "setup_v2.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n")
    temporary.replace(ASSETS / "setup_v2.json")
    print(f"Verified {len(checkpoints)} checkpoints and {len(normalizers)} normalizers.")
    print(f"Wrote {ASSETS / 'setup_v2.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
