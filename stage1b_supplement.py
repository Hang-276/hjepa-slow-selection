"""Stage-1B supplement on dev_preflight (no TEST role exists).

Per (model_tag, seed) it encodes train/val/audit once with a single ViT pass,
then optionally fits:
  * the official CLS probe at each requested level,
  * grid4x4_160k_official at Level 1,
  * grid4x4_160k_ridge at Level 1.

Run one checkpoint per process:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python -B stage1b_supplement.py --model-tag fourroom_lewm --seed 42 --no-grid-mlp
Then combine:
  CUDA_VISIBLE_DEVICES=6 .venv/bin/python -B stage1b_supplement.py --combine
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
LEVELS = {"fourroom_lewm": [1], "fourroom_hjepa_l2": [1, 2],
          "fourroom_hjepa_l4": [1, 2, 3, 4]}
OFFICIAL_CFG = {
    "hidden": 512,
    "epochs": 60,
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "drop_last": True,
    "target_columns": {"agent_xy": [0, 2], "distractor_xy": [2, 4]},
}
RIDGE_CFG = {
    "ridge_lambdas": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    "target_columns": {"agent_xy": [0, 2], "distractor_xy": [2, 4]},
}


def summarize(prediction, truth):
    r2 = run_v2.r2_per_axis(prediction, truth)
    return {"audit_r2": dict(zip(run_v2.AXES, r2.tolist())),
            "minimum_audit_r2": float(np.min(r2)),
            "distractor_r2_min": float(min(r2[2], r2[3]))}


def run_checkpoint(tag, seed, args):
    _, indices, _ = run_v2.load_splits(DATA / "splits_v2.json")
    model = run_v2.load_backbone({"model_tag": tag}, seed, args.device)
    with h5py.File(DATA / "fourroom.h5", "r") as data:
        sliced, grid, _ = run_v2.encode_roles_with_grid(
            model, data, indices, ROLES, args.device, 256, need_grid := (not args.no_grid_mlp or not args.no_ridge))
        truth = {role: run_v2.states(data, indices[role]) for role in ROLES}
    del model
    normalizer = run_v2.load_normalizer({"model_tag": tag}, seed)
    result = {"model_tag": tag, "seed": seed, "dataset": "dev_preflight",
              "levels": {}, "grid_mlp": None, "grid_ridge": None,
              "skipped": {"cls": args.no_cls, "grid_mlp": args.no_grid_mlp,
                          "ridge": args.no_ridge}}
    if not args.no_cls:
        for level in LEVELS[tag]:
            folder = OUT / f"stage1b_{tag}_seed{seed}_level{level}_cls"
            folder.mkdir(parents=True, exist_ok=True)
            artifact = run_v2.fit_probe_160k_official(
                sliced["train"][level - 1], truth["train"].reshape(-1, 4),
                sliced["val"][level - 1], truth["val"].reshape(-1, 4),
                normalizer, 42, OFFICIAL_CFG, args.device, folder)
            prediction = run_v2.predict_probe_160k_official(
                artifact, sliced["audit"][level - 1], args.device).reshape(
                    *indices["audit"].shape, 4)
            result["levels"][str(level)] = summarize(prediction, truth["audit"])
            logging.info("%s seed%s CLS L%s %s", tag, seed, level,
                         {k: round(v, 4) for k, v in result["levels"][str(level)]["audit_r2"].items()})
            del artifact
    if not args.no_grid_mlp:
        folder = OUT / f"stage1b_{tag}_seed{seed}_level1_grid_mlp"
        folder.mkdir(parents=True, exist_ok=True)
        artifact = run_v2.fit_probe_160k_official(
            grid["train"], truth["train"].reshape(-1, 4),
            grid["val"], truth["val"].reshape(-1, 4),
            normalizer, 42, OFFICIAL_CFG, args.device, folder)
        prediction = run_v2.predict_probe_160k_official(
            artifact, grid["audit"], args.device).reshape(*indices["audit"].shape, 4)
        result["grid_mlp"] = summarize(prediction, truth["audit"])
        logging.info("%s seed%s grid MLP L1 %s", tag, seed,
                     {k: round(v, 4) for k, v in result["grid_mlp"]["audit_r2"].items()})
        del artifact
    if not args.no_ridge:
        folder = OUT / f"stage1b_{tag}_seed{seed}_level1_grid_ridge"
        folder.mkdir(parents=True, exist_ok=True)
        artifact = run_v2.fit_probe_ridge(
            grid["train"], truth["train"].reshape(-1, 4),
            grid["val"], truth["val"].reshape(-1, 4),
            normalizer, RIDGE_CFG, args.device, folder)
        prediction = run_v2.predict_probe_ridge(
            artifact, grid["audit"], args.device).reshape(*indices["audit"].shape, 4)
        result["grid_ridge"] = summarize(prediction, truth["audit"])
        result["grid_ridge"]["selected_lambdas"] = {
            name: column["selected_lambda"] for name, column in artifact["columns"].items()}
        result["grid_ridge"]["validation_mse"] = {
            name: column["validation_mse"] for name, column in artifact["columns"].items()}
        logging.info("%s seed%s grid ridge L1 %s", tag, seed,
                     {k: round(v, 4) for k, v in result["grid_ridge"]["audit_r2"].items()})
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"stage1b_{tag}_seed{seed}.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    logging.info("Wrote %s", destination)
    return 0


def _prior_grid_mlp(tag, seed):
    candidates = [OUT / f"grid_readout_preflight_{tag}_seed{seed}.json"]
    if tag == "fourroom_hjepa_l2":
        candidates.append(OUT / f"grid_readout_preflight_seed{seed}.json")
    for path in candidates:
        if path.is_file():
            data = json.loads(path.read_text())
            value = data["features"]["grid4x4"]
            return {"audit_r2": value["audit_r2"],
                    "minimum_audit_r2": value["minimum_audit_r2"],
                    "distractor_r2_min": value.get("distractor_r2_min")}, path.name
    return None, None


def _prior_cls_levels(seed):
    path = OUT / f"dev_preflight_seed{seed}.json"
    if not path.is_file():
        return {}, None
    data = json.loads(path.read_text())
    levels = {}
    for level, value in data["levels"].items():
        r2 = value["audit_r2"]
        levels[level] = {
            "audit_r2": r2,
            "minimum_audit_r2": value["minimum_audit_r2"],
            "distractor_r2_min": value.get("distractor_r2_min", min(r2["distractor_x"], r2["distractor_y"])),
        }
    return levels, path.name


def combine():
    files = {}
    sources = {}
    for tag in LEVELS:
        for seed in (42, 43, 44):
            path = OUT / f"stage1b_{tag}_seed{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}")
            data = json.loads(path.read_text())
            used = {}
            if data.get("grid_mlp") is None:
                prior, source = _prior_grid_mlp(tag, seed)
                if prior is not None:
                    data["grid_mlp"] = prior
                    used["grid_mlp"] = source
            if not data["levels"] and tag == "fourroom_hjepa_l2":
                prior, source = _prior_cls_levels(seed)
                if prior:
                    data["levels"] = prior
                    used["cls"] = source
            files[(tag, seed)] = data
            sources[f"{tag}/seed{seed}"] = used
    gate = {}
    for tag in LEVELS:
        entry = {}
        for seed in (42, 43, 44):
            data = files[(tag, seed)]
            entry[str(seed)] = {
                "grid_mlp_min": (data["grid_mlp"] or {}).get("minimum_audit_r2"),
                "grid_ridge_min": (data["grid_ridge"] or {}).get("minimum_audit_r2"),
            }
        gate[tag] = entry
    seed44 = {tag: gate[tag]["44"] for tag in LEVELS}
    seed44_pass = {
        tag: any(value is not None and value >= 0.85
                 for value in (values["grid_mlp_min"], values["grid_ridge_min"]))
        for tag, values in seed44.items()
    }
    summary = {
        "gate_rule": "seed44 L1 grid4x4 (MLP and/or ridge) four-axis minimum >= 0.85 "
                     "for all three model families",
        "seed44": seed44,
        "seed44_pass": seed44_pass,
        "gate": "PASS_AUTO_PROCEED" if all(seed44_pass.values()) else "PAUSE_SEED44_BELOW_0.85",
        "reused_prior_sources": sources,
        "all": {tag: {str(seed): files[(tag, seed)] for seed in (42, 43, 44)} for tag in LEVELS},
    }
    destination = OUT / "stage1b_preflight_supplement.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"gate": summary["gate"], "seed44_pass": seed44_pass,
                      "seed44": seed44}, indent=2))
    return 0 if summary["gate"] == "PASS_AUTO_PROCEED" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-tag", choices=tuple(LEVELS))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--no-cls", action="store_true")
    parser.add_argument("--no-grid-mlp", action="store_true")
    parser.add_argument("--no-ridge", action="store_true")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.combine:
        return combine()
    if args.model_tag is None or args.seed is None:
        parser.error("--model-tag and --seed are required unless --combine is used")
    return run_checkpoint(args.model_tag, args.seed, args)


if __name__ == "__main__":
    sys.exit(main())
