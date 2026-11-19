"""Stage-1 dual-GPU preflight gate (no TEST role exists in this dataset).

Seeds 42 and 43 of fourroom_hjepa_l2 are encoded on the dev_preflight dataset
(400 train + 100 val full-frame trajectories, 64 audit windows) and probed with
probe_160k_official on levels 1 and 2. Each seed is meant to run on its own GPU:

  CUDA_VISIBLE_DEVICES=6 .venv/bin/python -B preflight_v2.py --seed 42
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python -B preflight_v2.py --seed 43

After both finish, --combine writes dev_preflight_summary.json and the gate
decision. The gate never authorizes reading any TEST data.
"""

import os

_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if _vis not in {"6", "7", "6,7", "7,6"}:
    raise RuntimeError(
        f"FATAL GPU ISOLATION GUARD: CUDA_VISIBLE_DEVICES={_vis!r}; only GPUs 6 and 7 are allowed!"
    )

import argparse
import json
import logging
from pathlib import Path
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

import run_v2

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "assets/data/dev_preflight"
OUT = ROOT / "results/preflight_audit"
ROLES = ("train", "val", "audit")
PROBE_CFG = {
    "hidden": 512,
    "epochs": 60,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "drop_last": True,
    "train_role": "train",
    "val_role": "val",
    "target_columns": {"agent_xy": [0, 2], "distractor_xy": [2, 4]},
}


def run_seed(seed, device):
    _, indices, _ = run_v2.load_splits(DATA / "splits_v2.json")
    if "test" in indices or set(indices) != set(ROLES):
        raise ValueError(f"Preflight splits must contain exactly {ROLES}; found {sorted(indices)}")
    receipt = json.loads((DATA / "data_receipt.json").read_text())
    if run_v2.sha256(DATA / "fourroom.h5") != receipt["h5_sha256"]:
        raise ValueError("dev_preflight h5 does not match its data receipt")
    if receipt.get("collection_seed") != 2026111800 or receipt.get("split_seed") != 20261117:
        raise ValueError("dev_preflight seeds are not the pre-registered ones")

    cfg = {"model_tag": "fourroom_hjepa_l2"}
    model = run_v2.load_backbone(cfg, seed, device)
    with h5py.File(DATA / "fourroom.h5", "r") as data:
        truth = {role: run_v2.states(data, indices[role]) for role in ROLES}
        sliced, role_lengths = run_v2.encode_roles(
            model, data, indices, ROLES, device, 256)
    del model

    normalizer = run_v2.load_normalizer(cfg, seed)
    results = {"seed": seed, "levels": {}, "roles": role_lengths,
               "data_receipt": receipt, "probe": PROBE_CFG}
    for level in (1, 2):
        folder = OUT / f"dev_preflight_seed{seed}_level{level}"
        folder.mkdir(parents=True, exist_ok=True)
        logging.info("seed %s level %s: fitting official 160k probe", seed, level)
        artifact = run_v2.fit_probe_160k_official(
            sliced["train"][level - 1], truth["train"].reshape(-1, 4),
            sliced["val"][level - 1], truth["val"].reshape(-1, 4),
            normalizer, 42, PROBE_CFG, device, folder)
        prediction = run_v2.predict_probe_160k_official(
            artifact, sliced["audit"][level - 1], device).reshape(*indices["audit"].shape, 4)
        r2 = run_v2.r2_per_axis(prediction, truth["audit"])
        np.savez_compressed(folder / "audit_prediction.npz",
                            prediction=prediction, truth=truth["audit"])
        results["levels"][str(level)] = {
            "audit_r2": dict(zip(run_v2.AXES, r2.tolist())),
            "minimum_audit_r2": float(np.min(r2)),
            "distractor_r2_min": float(min(r2[2], r2[3])),
            "eligible_at_0.85": bool(np.min(r2) >= 0.85),
        }
        logging.info("seed %s level %s AUDIT R2 %s", seed, level,
                     {axis: round(float(v), 4) for axis, v in zip(run_v2.AXES, r2)})
        del artifact

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"dev_preflight_seed{seed}.json"
    run_v2.save_json(path, results)
    logging.info("Wrote %s", path)
    return results


def combine():
    files = {seed: OUT / f"dev_preflight_seed{seed}.json" for seed in (42, 43)}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing preflight results: {missing}")
    results = {seed: json.loads(path.read_text()) for seed, path in files.items()}
    level1 = {seed: value["levels"]["1"] for seed, value in results.items()}
    both_min = {seed: level1[seed]["minimum_audit_r2"] for seed in (42, 43)}
    distractor = {seed: level1[seed]["distractor_r2_min"] for seed in (42, 43)}
    if any(value < 0.50 for value in distractor.values()):
        decision = "HARD_STOP_DISTRACTOR_R2_BELOW_0.50"
    elif all(value >= 0.85 for value in both_min.values()):
        decision = "PASS_AUTO_PROCEED"
    else:
        decision = "WARN_PROCEED_WITH_RECORDED_LEVEL1_R2"
    summary = {
        "gate": decision,
        "level1_min_audit_r2": both_min,
        "level1_distractor_r2_min": distractor,
        "level2_min_audit_r2": {seed: results[seed]["levels"]["2"]["minimum_audit_r2"]
                                for seed in (42, 43)},
        "per_seed": results,
        "note": "Preflight only. No TEST split exists in dev_preflight. "
                "AUTO_PROCEED follows the Stage-1 gate in the Wave-1 protocol.",
    }
    run_v2.save_json(OUT / "dev_preflight_summary.json", summary)
    print(json.dumps({k: summary[k] for k in
                      ("gate", "level1_min_audit_r2", "level1_distractor_r2_min")},
                     indent=2))
    return 0 if decision != "HARD_STOP_DISTRACTOR_R2_BELOW_0.50" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 43))
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.combine:
        return combine()
    if args.seed is None:
        parser.error("--seed or --combine is required")
    run_seed(args.seed, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
