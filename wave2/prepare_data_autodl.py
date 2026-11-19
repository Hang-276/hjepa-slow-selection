"""Single-GPU AutoDL data preparation for the shared fourroom_v2 dataset.

This is a guard-free adaptation of Wave-1 prepare_data_v2.py. It preserves the
exact formal collection and split logic:

  collection_seed = 2026111801, split_seed = 20261118
  train 400 x 401, val 100 x 401, audit 128 x 193, ref 256 x 193,
  cal 1024 x 193, test 2048 x 193; action_repeat_prob = 0.0.

The resulting splits_v2.json must hash to
a3f5d768564f0d49f8f99217bfc59ac30d0bff7672d4ba4c7093f05b708fbdb3.
If assets/data/fourroom_v2/data_receipt.json already verifies, collection is
skipped cleanly.

    python3 wave2/prepare_data_autodl.py --stage formal
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/H-JEPA"
FRAMES_PER_EPISODE = 401
WINDOW_LENGTH = 193
FULL_FRAME_ROLES = ("train", "val")
EXPECTED_SPLITS_SHA256 = "a3f5d768564f0d49f8f99217bfc59ac30d0bff7672d4ba4c7093f05b708fbdb3"

FORMAL = {
    "name": "fourroom_v2",
    "collection_seed": 2026111801,
    "split_seed": 20261118,
    "episodes": {"train": 400, "val": 100, "audit": 128,
                 "ref": 256, "cal": 1024, "test": 2048},
    "small": {"train_small_8k": (256, 32, "train"),
              "val_small_2k": (64, 32, "val")},
}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(data_dir, collection_out, cfg):
    data_dir.mkdir(parents=True, exist_ok=False)
    total = sum(cfg["episodes"].values())
    command = [
        sys.executable,
        str(SOURCE / "scripts/data/collect_fourroom_distractors.py"),
        f"cache_dir={data_dir}",
        "dataset_name=fourroom",
        f"seed={cfg['collection_seed']}",
        "num_traj=null",
        f"num_transitions={400 * total}",
        "ego_policy.mixture.0.params.action_repeat_prob=0.0",
        f"hydra.run.dir={collection_out}",
        "hydra.job.chdir=false",
    ]
    save_json(data_dir / "collection_command.json", command)
    env = dict(os.environ, PYTHONPATH=str(SOURCE), SDL_VIDEODRIVER="dummy")
    logging.info("Generating %s fresh episodes (%s transitions)", total, 400 * total)
    with (data_dir / "collection.log").open("w") as log:
        subprocess.run(command, cwd=SOURCE, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    collection_yaml = data_dir / "fourroom.collection.yaml"
    if not collection_yaml.is_file():
        raise FileNotFoundError(f"Collector did not write {collection_yaml}")
    shutil.copy2(collection_yaml, data_dir / "collection.yaml")
    return data_dir / "fourroom.h5"


def build_splits(data, cfg):
    lengths = data["ep_len"][:]
    offsets = data["ep_offset"][:]
    total = sum(cfg["episodes"].values())
    if len(lengths) != total or np.any(lengths != FRAMES_PER_EPISODE):
        raise ValueError(
            f"Expected exactly {total} fresh {FRAMES_PER_EPISODE}-frame episodes, "
            f"got {len(lengths)} episodes with lengths {sorted(set(lengths.tolist()))}")
    if data["pixels"].shape[1:] != (224, 224, 3):
        raise ValueError("Expected official 224x224 RGB frames")

    rng = np.random.default_rng(cfg["split_seed"])
    order = rng.permutation(len(lengths))
    roles, cursor = {}, 0
    for role, count in cfg["episodes"].items():
        ids = np.sort(order[cursor:cursor + count])
        cursor += count
        if role in FULL_FRAME_ROLES:
            local = np.broadcast_to(np.arange(FRAMES_PER_EPISODE),
                                    (count, FRAMES_PER_EPISODE))
        else:
            starts = rng.integers(0, FRAMES_PER_EPISODE + 1 - WINDOW_LENGTH, size=count)
            local = starts[:, None] + np.arange(WINDOW_LENGTH)
        roles[role] = {
            "episode_ids": ids.tolist(),
            "frame_indices": (offsets[ids, None] + local).tolist(),
        }
    if cursor != len(lengths):
        raise ValueError("Role counts do not cover every collected episode")

    small = {}
    for name, (episode_count, frames_per_episode, source_role) in cfg["small"].items():
        role = roles[source_role]
        ids = np.asarray(role["episode_ids"][:episode_count], dtype=np.int64)
        if len(ids) != episode_count:
            raise ValueError(f"Role {source_role} has too few episodes for {name}")
        small_rng = np.random.default_rng(cfg["split_seed"] + 1)
        local = np.stack([
            np.sort(small_rng.choice(FRAMES_PER_EPISODE, frames_per_episode, replace=False))
            for _ in ids
        ])
        small[name] = {
            "source_role": source_role,
            "episode_ids": ids.tolist(),
            "frame_indices": (offsets[ids, None] + local).tolist(),
            "description": f"first {episode_count} ascending {source_role} episode ids, "
                           f"{frames_per_episode} frames each",
        }
    return {
        "dataset_name": cfg["name"],
        "collection_seed": cfg["collection_seed"],
        "split_seed": cfg["split_seed"],
        "frames_per_episode": FRAMES_PER_EPISODE,
        "window_length": WINDOW_LENGTH,
        "roles": roles,
        "small": small,
    }


def verified_existing(data_dir):
    receipt_path = data_dir / "data_receipt.json"
    h5_path = data_dir / "fourroom.h5"
    splits_path = data_dir / "splits_v2.json"
    if not (receipt_path.is_file() and h5_path.is_file() and splits_path.is_file()):
        return False
    receipt = json.loads(receipt_path.read_text())
    ok = (sha256(h5_path) == receipt["h5_sha256"]
          and sha256(splits_path) == receipt["splits_sha256"]
          and receipt["splits_sha256"] == EXPECTED_SPLITS_SHA256)
    return bool(ok)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("formal",), default="formal")
    args = parser.parse_args()
    cfg = FORMAL
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not (SOURCE / ".git").is_dir():
        raise FileNotFoundError("Official H-JEPA source is missing; run setup_autodl.sh")
    data_dir = ROOT / "assets/data" / cfg["name"]
    if verified_existing(data_dir):
        logging.info("Existing %s verifies; skipping collection.", data_dir)
        return 0
    if data_dir.exists():
        raise FileExistsError(
            f"{data_dir} exists but does not verify; refusing to delete it. "
            "Move it aside manually if a fresh collection is intended.")

    h5_path = collect(data_dir, data_dir / "collection", cfg)
    with h5py.File(h5_path, "r") as data:
        splits = build_splits(data, cfg)
    splits_path = data_dir / "splits_v2.json"
    save_json(splits_path, splits)
    actual = sha256(splits_path)
    if actual != EXPECTED_SPLITS_SHA256:
        raise ValueError(
            f"Split hash {actual} does not match the pre-registered "
            f"{EXPECTED_SPLITS_SHA256}; the split logic is not identical.")

    receipt = {
        "dataset_name": cfg["name"],
        "stage": args.stage,
        "collection_seed": cfg["collection_seed"],
        "split_seed": cfg["split_seed"],
        "episodes": cfg["episodes"],
        "h5_path": str(h5_path.relative_to(ROOT)),
        "h5_bytes": h5_path.stat().st_size,
        "h5_sha256": sha256(h5_path),
        "splits_path": str(splits_path.relative_to(ROOT)),
        "splits_sha256": actual,
        "collection_yaml_sha256": sha256(data_dir / "collection.yaml"),
        "small_subsets": list(cfg["small"]),
    }
    save_json(data_dir / "data_receipt.json", receipt)
    logging.info("Wrote %s (%s bytes) and %s", h5_path.name,
                 receipt["h5_bytes"], splits_path.name)
    logging.info("h5 sha256=%s", receipt["h5_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
