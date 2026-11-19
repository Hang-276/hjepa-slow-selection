"""Collect the Wave-1 FourRoom datasets and freeze their splits.

Stage 1 uses --stage preflight: 400 train + 100 val full-frame trajectories and
64 audit windows, with seeds completely disjoint from fourroom_01 and from the
formal v2 dataset. No ref/cal/test roles exist at this stage.

Stage 2 uses --stage formal: 400 train + 100 val full-frame trajectories plus
audit/ref/cal/test windows, 3,956 trajectories in total.

The split logic is a faithful extension of run.py: whole episodes are permuted,
then assigned to roles; train/val keep all 401 frames, window roles get one
random 193-frame window. The 8k/2k small-readout subsets are pre-declared here.
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
import logging
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import hdf5plugin  # Registers the official collector's compression filter.
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "assets/H-JEPA"
FRAMES_PER_EPISODE = 401
WINDOW_LENGTH = 193
FULL_FRAME_ROLES = ("train", "val")

STAGES = {
    "preflight": {
        "name": "dev_preflight",
        "collection_seed": 2026111800,
        "split_seed": 20261117,
        "episodes": {"train": 400, "val": 100, "audit": 64},
        "small": {"train_small_8k": (256, 32, "train"), "val_small_2k": (64, 32, "val")},
    },
    "formal": {
        "name": "fourroom_v2",
        "collection_seed": 2026111801,
        "split_seed": 20261118,
        "episodes": {"train": 400, "val": 100, "audit": 128, "ref": 256, "cal": 1024, "test": 2048},
        "small": {"train_small_8k": (256, 32, "train"), "val_small_2k": (64, 32, "val")},
    },
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
        subprocess.run(command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
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
            f"got {len(lengths)} episodes with lengths {sorted(set(lengths.tolist()))}"
        )
    if data["pixels"].shape[1:] != (224, 224, 3):
        raise ValueError("Expected official 224x224 RGB frames")

    rng = np.random.default_rng(cfg["split_seed"])
    order = rng.permutation(len(lengths))
    roles, cursor = {}, 0
    for role, count in cfg["episodes"].items():
        ids = np.sort(order[cursor:cursor + count])
        cursor += count
        if role in FULL_FRAME_ROLES:
            local = np.broadcast_to(np.arange(FRAMES_PER_EPISODE), (count, FRAMES_PER_EPISODE))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    args = parser.parse_args()
    cfg = STAGES[args.stage]

    if not (SOURCE / ".git").is_dir():
        raise FileNotFoundError("Official H-JEPA source is missing")
    data_dir = ROOT / "assets/data" / cfg["name"]
    if data_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset directory {data_dir}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    h5_path = collect(data_dir, data_dir / "collection", cfg)
    with h5py.File(h5_path, "r") as data:
        splits = build_splits(data, cfg)
    splits_path = data_dir / "splits_v2.json"
    save_json(splits_path, splits)

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
        "splits_sha256": sha256(splits_path),
        "collection_yaml_sha256": sha256(data_dir / "collection.yaml"),
        "small_subsets": list(cfg["small"]),
    }
    save_json(data_dir / "data_receipt.json", receipt)
    logging.info("Wrote %s (%s bytes) and %s", h5_path.name, receipt["h5_bytes"], splits_path.name)
    logging.info("h5 sha256=%s", receipt["h5_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
