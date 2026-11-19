"""Wave-1 FourRoom H-JEPA / LeWM extension evaluation.

Three pre-registered arms share one freshly collected dataset:
  * fourroom_probe_v2     L=2, paired readout ablation (8k vs official 160k)
  * fourroom_lewm_v2      L=1 flat LeWM baseline
  * fourroom_hjepa_l4_v2  L=4 depth lineage, levels 1..4

The Pre-Test Forecast Lock is enforced structurally:
  (a) probes, REF variances and thresholds.json are frozen using only
      train/val/audit/ref/cal; CAL thresholds use predicted gaps only;
  (b) TEST pixels are encoded and every test_prediction.npz is written,
      then forecast_receipt.json is sealed;
  (c) only after the receipt exists may TEST physical coordinates be read.

Scoring and risk certification call the untouched metrics.py functions.
"""

import os

_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if _vis not in {"6", "7", "6,7", "7,6"}:
    raise RuntimeError(
        f"FATAL GPU ISOLATION GUARD: CUDA_VISIBLE_DEVICES={_vis!r}; only GPUs 6 and 7 are allowed!"
    )

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import hdf5plugin  # Registers the official collector's compression filter.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from metrics import risk, score_windows

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "assets/H-JEPA"
AXES = ["agent_x", "agent_y", "distractor_x", "distractor_y"]
NONTEST_ROLES = ("train", "val", "audit", "ref", "cal")
ALL_ROLES = NONTEST_ROLES + ("test",)
MODEL_TAGS = {"fourroom_lewm": 1, "fourroom_hjepa_l2": 2, "fourroom_hjepa_l4": 4}
SUBSET_NAME = {"train": "train_small_8k", "val": "val_small_2k"}
READOUT_INFO = {
    "cls_8k_small": {"feature": "cls", "probe": "8k"},
    "cls_160k_official": {"feature": "cls", "probe": "official"},
    "grid4x4_160k_official": {"feature": "grid", "probe": "official"},
    "grid4x4_160k_ridge": {"feature": "grid", "probe": "ridge"},
}


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_binary():
    found = shutil.which("git")
    if found:
        return found
    fallback = Path("/home/kemove/yuzihang/envs/git/bin/git")
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError("git is required to verify the official upstream checkout")


def relative_to_root(path):
    return str(Path(path).resolve().relative_to(ROOT))


def upstream_paths():
    for path in (str(SOURCE / "h_jepa"), str(SOURCE)):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    tag = cfg["model_tag"]
    if tag not in MODEL_TAGS:
        raise ValueError(f"Unsupported model_tag {tag!r}")
    if sorted(cfg["levels"]) != list(range(1, MODEL_TAGS[tag] + 1)):
        raise ValueError(f"{tag} requires levels {list(range(1, MODEL_TAGS[tag] + 1))}")
    if "level_readouts" in cfg:
        level_readouts = {int(level): list(readouts)
                          for level, readouts in cfg["level_readouts"].items()}
    else:
        level_readouts = {level: list(cfg["readouts"]) for level in cfg["levels"]}
    if sorted(level_readouts) != list(cfg["levels"]):
        raise ValueError("level_readouts must cover exactly the configured levels")
    for level, readouts in level_readouts.items():
        if not set(readouts) <= set(READOUT_INFO):
            raise ValueError(f"Unknown readout at level {level}: {readouts}")
        if level != 1 and any(READOUT_INFO[name]["feature"] == "grid" for name in readouts):
            raise ValueError("Spatial-grid readouts are defined for Level 1 only")
    cfg["_level_readouts"] = level_readouts
    cfg["_path"] = str(Path(path).resolve())
    return cfg


def load_splits(path):
    raw = json.loads(Path(path).read_text())
    roles = {role: np.asarray(value["frame_indices"], dtype=np.int64)
             for role, value in raw["roles"].items()}
    small = {name: np.asarray(value["frame_indices"], dtype=np.int64)
             for name, value in raw.get("small", {}).items()}
    return raw, roles, small


def states(data, indices):
    # Contiguous reads stay within one assigned episode.
    values = []
    for row in np.asarray(indices):
        start, stop = int(row[0]), int(row[-1]) + 1
        values.append(np.concatenate(
            [data["xy"][start:stop], data["distractor0_xy"][start:stop]], axis=1)[row - start])
    return np.stack(values).astype(np.float64)


def check_data_receipt(cfg):
    receipt = json.loads((ROOT / cfg["data"]["receipt"]).read_text())
    h5_path = ROOT / cfg["data"]["h5"]
    splits_path = ROOT / cfg["data"]["splits"]
    if sha256(h5_path) != receipt["h5_sha256"]:
        raise ValueError(f"{h5_path} no longer matches data_receipt.json")
    if sha256(splits_path) != receipt["splits_sha256"]:
        raise ValueError(f"{splits_path} no longer matches data_receipt.json")
    return receipt


# --------------------------------------------------------------------------- #
# Backbone loading and frozen encoding
# --------------------------------------------------------------------------- #

def load_backbone(cfg, seed, device):
    upstream_paths()
    tag = cfg["model_tag"]
    path = ROOT / f"assets/weights/fourroom/{tag}/seed{seed}/{tag}_object.ckpt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {path}; run download_extra_weights.py")
    model = torch.load(path, map_location="cpu", weights_only=False).eval().to(device)
    model.requires_grad_(False)
    expected = MODEL_TAGS[tag]
    if model.num_levels != expected:
        raise ValueError(f"{tag} has {model.num_levels} levels, expected {expected}")
    for level in range(1, model.num_levels + 1):
        jepa = model.get_level(level)
        if int(getattr(jepa, "temporal_window_size", 1)) != 1:
            raise ValueError(f"{tag} level {level} does not have temporal_window_size=1")
        dim = int(model.embed_dims[level])
        if dim != 192:
            raise ValueError(f"{tag} level {level} embed dim is {dim}, expected 192")
    return model


def read_pixels(dataset, batch):
    """Read a strictly increasing frame batch as contiguous slices.

    h5py falls back to a very slow point selection for arbitrary index arrays
    even when they are increasing; every role here is a union of contiguous
    episode ranges, so splitting at gaps restores slice-level throughput.
    """
    batch = np.asarray(batch, dtype=np.int64)
    if batch.size == 0 or (batch.size > 1 and not np.all(np.diff(batch) > 0)):
        raise ValueError("Pixel frame batch must be nonempty and strictly increasing")
    cuts = np.flatnonzero(np.diff(batch) != 1)
    starts = np.r_[0, cuts + 1]
    stops = np.r_[cuts + 1, batch.size]
    parts = [dataset[int(batch[s]):int(batch[e - 1]) + 1] for s, e in zip(starts, stops)]
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


@torch.inference_mode()
def encode_frames(model, data, frame_indices, device, batch_size):
    """Dense per-frame encode through every level, matching _dense_encode_no_stride."""
    frames = np.asarray(frame_indices, dtype=np.int64).ravel()
    result = [[] for _ in range(model.num_levels)]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    started = time.time()
    for start in range(0, len(frames), batch_size):
        batch = frames[start:start + batch_size]
        pixels = torch.from_numpy(np.ascontiguousarray(read_pixels(data["pixels"], batch)))
        pixels = pixels.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255
        embedded = model.get_level(1).encode({"pixels": ((pixels - mean) / std)[:, None]},
                                             key="pixels")["embed_0"]
        result[0].append(embedded[:, 0].cpu().numpy())
        for level in range(2, model.num_levels + 1):
            key = f"embed_{level - 1}"
            embedded = model.get_level(level).encode({key: embedded}, key=key)["embed_0"]
            result[level - 1].append(embedded[:, 0].cpu().numpy())
        if start and start % (batch_size * 200) == 0:
            rate = start / max(time.time() - started, 1e-9)
            logging.info("Encoded %s / %s frames (%.0f frames/s)", start, len(frames), rate)
    output = [np.concatenate(part).astype(np.float32, copy=False) for part in result]
    if any(not np.isfinite(level).all() for level in output):
        raise ValueError("Nonfinite backbone features")
    return output


def encode_roles(model, data, indices, roles, device, batch_size):
    # Roles are encoded separately: h5py requires strictly increasing fancy
    # indices, and concatenating two roles would interleave episode offsets.
    role_order = list(roles)
    lengths, per_role = {}, {}
    for role in role_order:
        frames = np.asarray(indices[role], dtype=np.int64).ravel()
        if frames.size and not np.all(np.diff(frames) > 0):
            raise ValueError(f"Frame indices for role {role} are not strictly increasing")
        per_role[role] = encode_frames(model, data, frames, device, batch_size)
        lengths[role] = int(frames.size)
    num_levels = len(per_role[role_order[0]])
    sliced = {
        role: [per_role[role][level] for level in range(num_levels)]
        for role in role_order
    }
    return sliced, lengths


def _level1_cls_and_grid(model, normalized_pixels):
    """One Level-1 ViT pass: official CLS output plus the 4x4 spatial grid.

    The grid is the projector-applied patch tokens reshaped to 16x16, mean-pooled
    into 4x4 blocks, and flattened to 3072-D. The CLS path is exactly the one
    ProjectedEncoder.forward uses.
    """
    encoder = model.get_level(1).encoder
    if not (hasattr(encoder, "encoder") and hasattr(encoder, "projector")):
        raise ValueError("Level-1 encoder is not a ProjectedEncoder")
    hidden = encoder.encoder(normalized_pixels,
                             interpolate_pos_encoding=True).last_hidden_state
    if hidden.shape[1] != 257 or hidden.shape[-1] != 192:
        raise ValueError(
            f"Expected CLS + 256 patch tokens of width 192, got {tuple(hidden.shape)}")
    cls = encoder.projector(hidden[:, 0])
    patches = hidden[:, 1:]
    projected = encoder.projector(
        patches.reshape(-1, patches.shape[-1])).reshape(patches.shape)
    grid = projected.reshape(-1, 16, 16, projected.shape[-1])
    grid = grid.reshape(-1, 4, 4, 4, 4, projected.shape[-1]).mean(dim=(2, 4))
    return cls, grid.reshape(grid.shape[0], -1)


@torch.inference_mode()
def encode_frames_with_grid(model, data, frame_indices, device, batch_size, want_grid):
    """Dense encode returning per-level CLS features and optionally the L1 grid.

    A single ViT forward pass yields both outputs, so the HDF5 pixels are read
    exactly once per checkpoint.
    """
    frames = np.asarray(frame_indices, dtype=np.int64).ravel()
    result = [[] for _ in range(model.num_levels)]
    grid_result = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    started = time.time()
    for start in range(0, len(frames), batch_size):
        batch = frames[start:start + batch_size]
        pixels = torch.from_numpy(np.ascontiguousarray(read_pixels(data["pixels"], batch)))
        pixels = pixels.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255
        normalized = (pixels - mean) / std
        cls, grid = _level1_cls_and_grid(model, normalized)
        embedded = cls[:, None, :]
        result[0].append(cls.cpu().numpy())
        if want_grid:
            grid_result.append(grid.cpu().numpy())
        for level in range(2, model.num_levels + 1):
            key = f"embed_{level - 1}"
            embedded = model.get_level(level).encode({key: embedded}, key=key)["embed_0"]
            result[level - 1].append(embedded[:, 0].cpu().numpy())
        if start and start % (batch_size * 200) == 0:
            rate = start / max(time.time() - started, 1e-9)
            logging.info("Encoded %s / %s frames (%.0f frames/s)", start, len(frames), rate)
    output = [np.concatenate(part).astype(np.float32, copy=False) for part in result]
    grid_output = (np.concatenate(grid_result).astype(np.float32, copy=False)
                   if want_grid else None)
    if any(not np.isfinite(level).all() for level in output):
        raise ValueError("Nonfinite backbone features")
    if grid_output is not None and not np.isfinite(grid_output).all():
        raise ValueError("Nonfinite grid features")
    return output, grid_output


def encode_roles_with_grid(model, data, indices, roles, device, batch_size, want_grid):
    """Role-wise encode returning per-level CLS slices and optional L1 grid slices."""
    role_order = list(roles)
    lengths, cls_per_role, grid_per_role = {}, {}, {}
    for role in role_order:
        frames = np.asarray(indices[role], dtype=np.int64).ravel()
        if frames.size and not np.all(np.diff(frames) > 0):
            raise ValueError(f"Frame indices for role {role} are not strictly increasing")
        cls, grid = encode_frames_with_grid(model, data, frames, device, batch_size, want_grid)
        cls_per_role[role] = cls
        if want_grid:
            grid_per_role[role] = grid
        lengths[role] = int(frames.size)
    num_levels = len(cls_per_role[role_order[0]])
    sliced = {
        role: [cls_per_role[role][level] for level in range(num_levels)]
        for role in role_order
    }
    return sliced, (grid_per_role if want_grid else None), lengths


# --------------------------------------------------------------------------- #
# Frozen feature cache
# --------------------------------------------------------------------------- #

def cache_root(cfg):
    return ROOT / cfg["feature_cache"]["root"]


def feature_cache_paths(cfg, seed, suffix):
    root = cache_root(cfg)
    stem = f"{cfg['model_tag']}_seed{seed}"
    return root / f"{stem}_{suffix}.npy", root / f"{stem}_features.json"


def _sidecar_update(path, entry_name, entry):
    current = json.loads(path.read_text()) if path.exists() else {"files": {}}
    current.setdefault("files", {})[entry_name] = entry
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(current, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cache_enabled(cfg):
    if not cfg["feature_cache"].get("enabled", True):
        logging.info("Feature cache disabled by config")
        return False
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    minimum = float(cfg["feature_cache"].get("min_free_gb", 30))
    if free_gb < minimum:
        logging.warning("Feature cache disabled: only %.1f GB free (< %.1f GB)", free_gb, minimum)
        return False
    root = cache_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    return True


def save_or_reuse_features(cfg, seed, suffix, features, role_lengths, checkpoint_sha):
    """Save [levels, frames, 192] float32 features, or reuse a hash-verified cache."""
    path, sidecar = feature_cache_paths(cfg, seed, suffix)
    if path.exists() and sidecar.exists():
        record = json.loads(sidecar.read_text()).get("files", {}).get(suffix)
        if record and record.get("checkpoint_sha256") == checkpoint_sha and sha256(path) == record.get("sha256"):
            cached = np.load(path)
            if cached.dtype != np.float32 or cached.shape != (len(features), sum(role_lengths.values()), 192):
                raise ValueError(f"Cached feature shape/dtype mismatch for {path}")
            logging.info("Reusing hashed feature cache %s", path.name)
            return cached
        raise ValueError(f"Feature cache {path} exists but does not match this checkpoint")
    stacked = np.stack(features).astype(np.float32, copy=False)
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, stacked)
    temporary.replace(path)
    _sidecar_update(sidecar, suffix, {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "shape": list(stacked.shape),
        "dtype": "float32",
        "role_frames": role_lengths,
        "checkpoint_sha256": checkpoint_sha,
    })
    return stacked


# --------------------------------------------------------------------------- #
# Readouts
# --------------------------------------------------------------------------- #

def make_joint_probe(dimension, hidden, device):
    # Identical to run.py: two 128-D SiLU hidden layers, one joint 4-D head.
    return nn.Sequential(nn.Linear(dimension, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4)).to(device)


def fit_probe_8k(train_x, train_y, val_x, val_y, seed, pcfg, device, folder):
    torch.manual_seed(seed)
    x_mean, x_std = train_x.mean(0), np.maximum(train_x.std(0), 1e-6)
    y_mean, y_std = train_y.mean(0), train_y.std(0)
    if np.any(y_std <= 0):
        raise ValueError("A training physical factor has zero variance")
    x = torch.as_tensor((train_x - x_mean) / x_std, device=device)
    y = torch.as_tensor((train_y - y_mean) / y_std, dtype=torch.float32, device=device)
    vx = torch.as_tensor((val_x - x_mean) / x_std, device=device)
    vy = torch.as_tensor((val_y - y_mean) / y_std, dtype=torch.float32, device=device)
    probe = make_joint_probe(x.shape[1], pcfg["hidden"], device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=pcfg["learning_rate"],
                                  weight_decay=pcfg["weight_decay"])
    best_loss, best, trace = float("inf"), None, []
    for step in range(1, pcfg["steps"] + 1):
        ids = torch.randint(len(x), (pcfg["batch_size"],), device=device)
        loss = (probe(x[ids]) - y[ids]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite probe training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == pcfg["steps"]:
            with torch.no_grad():
                val_loss = (probe(vx) - vy).square().mean().item()
            trace.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss})
            if val_loss < best_loss:
                best_loss = val_loss
                best = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
    artifact = {"format": "wave1_probe_8k_v1", "state_dict": best, "input_dim": int(x.shape[1]),
                "hidden": pcfg["hidden"], "x_mean": torch.from_numpy(x_mean),
                "x_std": torch.from_numpy(x_std), "y_mean": torch.from_numpy(y_mean),
                "y_std": torch.from_numpy(y_std)}
    torch.save(artifact, folder / "probe_8k.pt")
    save_json(folder / "probe_8k.json", {"seed": seed, "validation_loss": best_loss,
                                         "trace": trace, "steps": pcfg["steps"],
                                         "train_frames": int(len(x)), "val_frames": int(len(vx))})
    return artifact


@torch.inference_mode()
def predict_probe_8k(artifact, features, device):
    probe = make_joint_probe(artifact["input_dim"], artifact["hidden"], device)
    probe.load_state_dict(artifact["state_dict"])
    probe.eval()
    mean, std = artifact["x_mean"].numpy(), artifact["x_std"].numpy()
    output = []
    for chunk in range(0, len(features), 16384):
        x = torch.as_tensor((features[chunk:chunk + 16384] - mean) / std, device=device)
        output.append(probe(x).cpu().numpy())
    return np.concatenate(output) * artifact["y_std"].numpy() + artifact["y_mean"].numpy()


def load_normalizer(cfg, seed):
    path = ROOT / f"assets/weights/fourroom/{cfg['model_tag']}/seed{seed}/normalizer.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing normalizer {path}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    stats = artifact["stats"]
    return {
        "agent_xy": (np.asarray(stats["xy"]["mean"], dtype=np.float32).reshape(-1),
                     np.asarray(stats["xy"]["std"], dtype=np.float32).reshape(-1)),
        "distractor_xy": (np.asarray(stats["distractor0_xy"]["mean"], dtype=np.float32).reshape(-1),
                          np.asarray(stats["distractor0_xy"]["std"], dtype=np.float32).reshape(-1)),
    }


def fit_probe_160k_official(train_x, train_y, val_x, val_y, norm_stats, seed, pcfg,
                            device, folder):
    """Audited upstream probe: raw 192-D input, one 512-D LN+GELU hidden layer per
    target column, official normalizer target scaling, AdamW, 60 epochs,
    shuffle+drop_last, no scheduler, NO validation-based selection (final epoch)."""
    upstream_paths()
    from models.module import MLP  # noqa: PLC0415

    x_train = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    x_val = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    columns = list(pcfg["target_columns"])
    means, stds, targets, val_targets, heads = {}, {}, {}, {}, {}
    torch.manual_seed(seed)
    for name in columns:
        lo, hi = pcfg["target_columns"][name]
        mean, std = norm_stats[name]
        if mean.size != hi - lo or std.size != hi - lo:
            raise ValueError(f"Normalizer width mismatch for {name}")
        means[name], stds[name] = mean, std
        targets[name] = torch.as_tensor((train_y[:, lo:hi] - mean) / std,
                                        dtype=torch.float32, device=device)
        val_targets[name] = torch.as_tensor((val_y[:, lo:hi] - mean) / std,
                                            dtype=torch.float32, device=device)
        heads[name] = MLP(input_dim=x_train.shape[1], hidden_dim=pcfg["hidden"],
                          output_dim=hi - lo).to(device)

    parameters = [p for head in heads.values() for p in head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=pcfg["learning_rate"],
                                  weight_decay=pcfg["weight_decay"])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    batch_size = int(pcfg["batch_size"])
    n = x_train.shape[0]
    batches = n // batch_size if pcfg.get("drop_last", True) else int(np.ceil(n / batch_size))
    trace = []
    for epoch in range(1, int(pcfg["epochs"]) + 1):
        for head in heads.values():
            head.train()
        permutation = torch.randperm(n, generator=generator, device=device)
        total = 0.0
        for batch in range(batches):
            ids = permutation[batch * batch_size:(batch + 1) * batch_size]
            loss = None
            for name, head in heads.items():
                term = F.mse_loss(head(x_train[ids]), targets[name][ids])
                loss = term if loss is None else loss + term
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        with torch.no_grad():
            val_losses = {}
            for name, head in heads.items():
                head.eval()
                val_losses[name] = float(F.mse_loss(head(x_val), val_targets[name]))
        trace.append({"epoch": epoch, "train_loss_sum": total / batches, **val_losses})
        if epoch == 1 or epoch % 10 == 0 or epoch == int(pcfg["epochs"]):
            logging.info("official probe epoch %s/%s train=%.6f val=%s", epoch,
                         pcfg["epochs"], total / batches,
                         {k: round(v, 6) for k, v in val_losses.items()})

    artifact = {
        "format": "wave1_probe_160k_official_v1",
        "input_dim": int(x_train.shape[1]),
        "hidden": int(pcfg["hidden"]),
        "columns": {},
        "seed": seed,
        "epochs": int(pcfg["epochs"]),
        "batch_size": batch_size,
        "learning_rate": float(pcfg["learning_rate"]),
        "weight_decay": float(pcfg["weight_decay"]),
        "drop_last": bool(pcfg.get("drop_last", True)),
        "scheduler": None,
        "selection": "final_epoch",
        "train_frames": int(x_train.shape[0]),
        "val_frames": int(x_val.shape[0]),
    }
    for name, head in heads.items():
        lo, hi = pcfg["target_columns"][name]
        artifact["columns"][name] = {
            "state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
            "target_slice": [int(lo), int(hi)],
            "target_mean": torch.from_numpy(means[name]),
            "target_std": torch.from_numpy(stds[name]),
        }
    torch.save(artifact, folder / "probe_160k_official.pt")
    save_json(folder / "probe_160k_official.json", {
        "seed": seed, "trace": trace, "train_frames": int(len(train_x)),
        "val_frames": int(len(val_x)), "final_val_losses": trace[-1],
        "note": "Official alignment audited in results/preflight_audit/upstream_probe_audit.json",
    })
    return artifact


@torch.inference_mode()
def predict_probe_160k_official(artifact, features, device):
    upstream_paths()
    from models.module import MLP  # noqa: PLC0415

    output = []
    for name, column in artifact["columns"].items():
        head = MLP(input_dim=artifact["input_dim"], hidden_dim=artifact["hidden"],
                   output_dim=column["target_slice"][1] - column["target_slice"][0]).to(device)
        head.load_state_dict(column["state_dict"])
        head.eval()
        mean = column["target_mean"].numpy()
        std = column["target_std"].numpy()
        parts = []
        for chunk in range(0, len(features), 16384):
            x = torch.as_tensor(features[chunk:chunk + 16384], device=device)
            parts.append(head(x).cpu().numpy() * std + mean)
        output.append(np.concatenate(parts))
    return np.concatenate(output, axis=1)


def fit_probe_ridge(train_x, train_y, val_x, val_y, normalizer, pcfg, device, folder):
    """Closed-form Ridge on standardized inputs, lambda selected on VAL only.

    Targets are normalized with the checkpoint normalizer, exactly like the
    official MLP probes. One lambda is chosen per target column by validation
    MSE; audit/cal/test are never consulted.
    """
    lambdas = [float(value) for value in pcfg["ridge_lambdas"]]
    if not lambdas or any(value <= 0 for value in lambdas):
        raise ValueError("ridge_lambdas must be positive")
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    xv = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    x_mean = x.mean(0)
    x_std = x.std(0).clamp_min(1e-6)
    xz = (x - x_mean) / x_std
    xzv = (xv - x_mean) / x_std
    gram = xz.T @ xz
    eye = torch.eye(gram.shape[0], device=device)
    artifact = {
        "format": "wave1_probe_grid4x4_ridge_v1",
        "input_dim": int(train_x.shape[1]),
        "x_mean": x_mean.detach().cpu(),
        "x_std": x_std.detach().cpu(),
        "ridge_lambdas": lambdas,
        "columns": {},
        "selection": "lambda_by_val_mse",
        "train_frames": int(train_x.shape[0]),
        "val_frames": int(val_x.shape[0]),
    }
    trace = {}
    for name, (lo, hi) in pcfg["target_columns"].items():
        mean, std = normalizer[name]
        target_mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        target_std = torch.as_tensor(std, dtype=torch.float32, device=device)
        y = (torch.as_tensor(train_y[:, lo:hi], dtype=torch.float32, device=device)
             - target_mean) / target_std
        yv = (torch.as_tensor(val_y[:, lo:hi], dtype=torch.float32, device=device)
              - target_mean) / target_std
        y_offset = y.mean(0)
        rhs = xz.T @ (y - y_offset)
        best_mse, best_lambda, best_weights, sweep = float("inf"), None, None, []
        for lam in lambdas:
            weights = torch.linalg.solve(gram + lam * eye, rhs)
            prediction = xzv @ weights + y_offset
            mse = float(((prediction - yv) ** 2).mean())
            if not np.isfinite(mse):
                raise ValueError("Nonfinite ridge validation MSE")
            sweep.append({"lambda": lam, "val_mse": mse})
            if mse < best_mse:
                best_mse, best_lambda, best_weights = mse, lam, weights
        if best_weights is None:
            raise ValueError("Ridge lambda sweep produced no solution")
        artifact["columns"][name] = {
            "target_slice": [int(lo), int(hi)],
            "weights": best_weights.detach().cpu(),
            "target_offset": y_offset.detach().cpu(),
            "target_mean": torch.from_numpy(mean),
            "target_std": torch.from_numpy(std),
            "selected_lambda": best_lambda,
            "validation_mse": best_mse,
        }
        trace[name] = {"sweep": sweep, "selected_lambda": best_lambda,
                       "validation_mse": best_mse}
        logging.info("ridge %s selected lambda=%g val_mse=%.6g", name, best_lambda, best_mse)
    torch.save(artifact, folder / "probe_grid4x4_ridge.pt")
    save_json(folder / "probe_grid4x4_ridge.json", trace)
    return artifact


@torch.inference_mode()
def predict_probe_ridge(artifact, features, device):
    x_mean = artifact["x_mean"].to(device)
    x_std = artifact["x_std"].to(device)
    output = []
    for name, column in artifact["columns"].items():
        weights = column["weights"].to(device)
        offset = column["target_offset"].to(device)
        mean = column["target_mean"].numpy()
        std = column["target_std"].numpy()
        parts = []
        for chunk in range(0, len(features), 16384):
            x = torch.as_tensor(features[chunk:chunk + 16384],
                                dtype=torch.float32, device=device)
            parts.append(((x - x_mean) / x_std @ weights + offset).cpu().numpy())
        output.append(np.concatenate(parts) * std + mean)
    return np.concatenate(output, axis=1)


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #

def r2_per_axis(prediction, truth):
    variance = truth.reshape(-1, truth.shape[-1]).var(0)
    if np.any(variance <= 0):
        raise ValueError("A reference factor has zero variance")
    return 1 - np.mean((prediction - truth) ** 2, axis=tuple(range(prediction.ndim - 1))) / variance


def risk_with(alpha, gap, wrong, threshold, cfg_eval):
    return risk(gap, wrong, threshold, alpha, cfg_eval["minimum_coverage"],
                cfg_eval["maximum_selective_error"])


def group_diagnostics(scores, threshold, alpha, cfg_eval):
    gap, wrong = scores["gap"], scores["wrong"]
    regular = (scores["true_top2"][:, 0] == 0) & (scores["true_top2"][:, 1] == 1)
    accept = gap >= threshold
    groups = {}
    for name, mask in (("regular_agent_slow", regular), ("local_flip", ~regular)):
        count = int(mask.sum())
        if count == 0:
            groups[name] = {"windows": 0}
            continue
        accepted = int((accept & mask).sum())
        errors = int((wrong & accept & mask).sum())
        groups[name] = {
            "windows": count,
            "full_accuracy": float(1 - wrong[mask].mean()),
            "accepted": accepted,
            "errors": errors,
            "coverage": accepted / count,
            "selective_error": errors / accepted if accepted else None,
        }
    return groups


def _quantiles(values):
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    return {str(q): np.quantile(values, q, axis=0).tolist() for q in qs}


# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #

def prepare_output(args, cfg):
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "config.json").open("x") as stream:
        json.dump({**cfg, "run_name": args.name, "device": args.device,
                   "cuda_visible_devices": _vis}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    return out


def snapshot_scripts(out, cfg):
    folder = out / "scripts"
    folder.mkdir()
    names = ["run_v2.py", "metrics.py", "prepare_data_v2.py", "download_extra_weights.py",
             "preflight_v2.py", "verify_v2.py", "stage1b_supplement.py",
             "combine_depth_v2.py"]
    copied = {}
    for name in names:
        source = ROOT / name
        if source.is_file():
            shutil.copy2(source, folder / name)
            copied[name] = sha256(source)
    config_source = Path(cfg["_path"])
    shutil.copy2(config_source, folder / config_source.name)
    copied[config_source.name] = sha256(config_source)
    save_json(out / "script_receipt.json", copied)


def check_provenance(cfg, device, out):
    git = git_binary()
    commit = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    if commit != cfg["upstream_commit"]:
        raise ValueError("Upstream source revision does not match the pinned commit")
    if subprocess.check_output([git, "status", "--porcelain", "--untracked-files=no"],
                               cwd=SOURCE, text=True):
        raise ValueError("Official tracked source has local changes")
    setup = json.loads((ROOT / "assets/setup_v2.json").read_text())
    weights = {}
    for seed in cfg["model_seeds"]:
        relative = f"{cfg['model_tag']}/seed{seed}"
        path = ROOT / "assets/weights" / setup["checkpoints"][relative]["relative_path"]
        digest = sha256(path)
        if digest != setup["checkpoints"][relative]["sha256"]:
            raise ValueError(f"Checkpoint hash differs from setup_v2.json: {path}")
        norm_relative = f"{cfg['model_tag']}/seed{seed}"
        norm_path = ROOT / "assets/weights" / setup["normalizers"][norm_relative]["relative_path"]
        if sha256(norm_path) != setup["normalizers"][norm_relative]["sha256"]:
            raise ValueError(f"Normalizer hash differs from setup_v2.json: {norm_path}")
        weights[seed] = {"path": relative_to_root(path), "sha256": digest}
    provenance = {
        "upstream_commit": commit,
        "weights_sha256": weights,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cuda_visible_devices": _vis,
        "setup_v2_sha256": sha256(ROOT / "assets/setup_v2.json"),
    }
    save_json(out / "provenance.json", provenance)
    subprocess.run([sys.executable, "-m", "pip", "freeze"],
                   stdout=(out / "environment.txt").open("w"), check=True)
    return provenance


def build_cells(cfg):
    return [(readout, seed, level)
            for level in cfg["levels"]
            for readout in cfg["_level_readouts"][level]
            for seed in cfg["model_seeds"]]


def cell_name(readout, seed, level):
    return f"{readout}_seed{seed}_level{level}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                            for c in args.name):
        parser.error("Use only letters, digits, '_' and '-' in --name")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        parser.error("Exactly one visible GPU is required per process")

    cfg = load_config(args.config)
    eval_cfg = cfg["eval"]
    alpha = eval_cfg["family_alpha"] / eval_cfg["family_size"]
    out = prepare_output(args, cfg)
    save_json(out / "status.json", {"status": "RUNNING"})
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["HF_HOME"] = str(ROOT / "assets/cache")

    try:
        snapshot_scripts(out, cfg)
        provenance = check_provenance(cfg, args.device, out)
        receipt = check_data_receipt(cfg)
        save_json(out / "data.json", receipt)
        splits_raw, indices, small_indices = load_splits(ROOT / cfg["data"]["splits"])
        for role in NONTEST_ROLES:
            if role not in indices:
                raise ValueError(f"Dataset {cfg['data']['name']} is missing role {role}")
        if "test" not in indices:
            raise ValueError("Formal run requires a sealed test role")
        cache_ok = cache_enabled(cfg)
        setup = json.loads((ROOT / "assets/setup_v2.json").read_text())

        expected_cells = len(build_cells(cfg))
        thresholds = {}
        level1_thresholds = {}
        reference_variance = {}
        audit_records = {}
        probes = {}

        with h5py.File(ROOT / cfg["data"]["h5"], "r") as data:
            physical_ref = states(data, indices["ref"])
            physical_ref_variance = physical_ref.reshape(-1, 4).var(axis=0)
            np.savez_compressed(out / "reference_truth.npz", states=physical_ref,
                                variance=physical_ref_variance)
            truth = {role: states(data, indices[role]) for role in NONTEST_ROLES}
            truth["train_small_8k"] = states(data, small_indices["train_small_8k"])
            truth["val_small_2k"] = states(data, small_indices["val_small_2k"])

            needs_grid = any(
                READOUT_INFO[readout]["feature"] == "grid"
                for level in cfg["levels"] for readout in cfg["_level_readouts"][level])

            for seed in cfg["model_seeds"]:
                logging.info("Model seed %s: loading %s", seed, cfg["model_tag"])
                model = load_backbone(cfg, seed, args.device)
                checkpoint_sha = setup["checkpoints"][f"{cfg['model_tag']}/seed{seed}"]["sha256"]
                sliced, grid_sliced, role_lengths = encode_roles_with_grid(
                    model, data, indices, NONTEST_ROLES, args.device,
                    cfg["encode_batch_size"], needs_grid)
                if cache_ok:
                    stacked = save_or_reuse_features(
                        cfg, seed, "nontest",
                        [np.concatenate([sliced[role][level] for role in NONTEST_ROLES])
                         for level in range(MODEL_TAGS[cfg["model_tag"]])],
                        role_lengths, checkpoint_sha)
                    offset = 0
                    for role in NONTEST_ROLES:
                        stop = offset + role_lengths[role]
                        for level in range(MODEL_TAGS[cfg["model_tag"]]):
                            sliced[role][level] = stacked[level][offset:stop]
                        offset = stop
                small, _ = encode_roles(model, data, small_indices, tuple(SUBSET_NAME.values()),
                                        args.device, cfg["encode_batch_size"])
                normalizer = load_normalizer(cfg, seed)

                for level in cfg["levels"]:
                    for readout in cfg["_level_readouts"][level]:
                        info = READOUT_INFO[readout]
                        cell = cell_name(readout, seed, level)
                        folder = out / cell
                        folder.mkdir()
                        logging.info("%s: fitting %s (%s/%s)", cell, readout,
                                     info["feature"], info["probe"])
                        if info["probe"] == "8k":
                            artifact = fit_probe_8k(
                                small["train_small_8k"][level - 1],
                                truth["train_small_8k"].reshape(-1, 4),
                                small["val_small_2k"][level - 1],
                                truth["val_small_2k"].reshape(-1, 4),
                                seed * 10 + level, cfg["probe_8k"], args.device, folder)
                            predict_fn = predict_probe_8k
                        elif info["probe"] == "official":
                            features = sliced if info["feature"] == "cls" else grid_sliced
                            index = level - 1 if info["feature"] == "cls" else None
                            train_x = features["train"][index] if index is not None else features["train"]
                            val_x = features["val"][index] if index is not None else features["val"]
                            artifact = fit_probe_160k_official(
                                train_x, truth["train"].reshape(-1, 4),
                                val_x, truth["val"].reshape(-1, 4),
                                normalizer, cfg["probe_seed"], cfg["probe_160k_official"],
                                args.device, folder)
                            predict_fn = predict_probe_160k_official
                        else:
                            if grid_sliced is None:
                                raise ValueError("Ridge readout requires grid features")
                            artifact = fit_probe_ridge(
                                grid_sliced["train"], truth["train"].reshape(-1, 4),
                                grid_sliced["val"], truth["val"].reshape(-1, 4),
                                normalizer, cfg["probe_grid4x4_ridge"], args.device, folder)
                            predict_fn = predict_probe_ridge
                        probes[cell] = (info["probe"], info["feature"], artifact)

                        def features_for(role):
                            if info["feature"] == "cls":
                                return sliced[role][level - 1]
                            if grid_sliced is None:
                                raise ValueError("Grid features were not extracted")
                            return grid_sliced[role]

                        audit_prediction = predict_fn(artifact, features_for("audit"), args.device)
                        audit_prediction = audit_prediction.reshape(*indices["audit"].shape, 4)
                        audit_truth = truth["audit"]
                        audit_r2 = r2_per_axis(audit_prediction, audit_truth)
                        audit_records[cell] = {
                            "r2": dict(zip(AXES, audit_r2.tolist())),
                            "eligible": bool(np.min(audit_r2) >= eval_cfg["audit_min_r2"]),
                        }
                        save_json(folder / "audit.json", audit_records[cell])
                        np.savez_compressed(folder / "audit.npz",
                                            prediction=audit_prediction, truth=audit_truth)

                        ref_prediction = predict_fn(artifact, features_for("ref"), args.device)
                        ref_prediction = ref_prediction.reshape(*indices["ref"].shape, 4)
                        ref_variance = ref_prediction.reshape(-1, 4).var(axis=0)
                        reference_variance[cell] = ref_variance
                        np.savez_compressed(folder / "reference.npz",
                                            prediction=ref_prediction, variance=ref_variance)

                        cal_prediction = predict_fn(artifact, features_for("cal"), args.device)
                        cal_prediction = cal_prediction.reshape(*indices["cal"].shape, 4)
                        # CAL truth is never loaded here: thresholding uses predicted gaps.
                        cal_scores = score_windows(cal_prediction, cal_prediction,
                                                   ref_variance, ref_variance, eval_cfg["lags"])
                        thresholds[cell] = float(np.quantile(cal_scores["gap"],
                                                             eval_cfg["cal_gap_quantile"]))
                        np.savez_compressed(folder / "calibration.npz",
                                            prediction=cal_prediction,
                                            scores=cal_scores["predicted_scores"],
                                            gap=cal_scores["gap"])
                        logging.info("%s AUDIT min R2=%.4f CAL threshold=%.6f", cell,
                                     float(np.min(audit_r2)), thresholds[cell])
                del model, sliced, grid_sliced, small

        all_readouts = {readout for level in cfg["levels"]
                        for readout in cfg["_level_readouts"][level]}
        level1_thresholds = {
            f"{readout}/seed{seed}": thresholds[cell_name(readout, seed, 1)]
            for readout in all_readouts for seed in cfg["model_seeds"]
            if cell_name(readout, seed, 1) in thresholds
        }
        grid_level1_thresholds = {
            f"seed{seed}": thresholds.get(cell_name("grid4x4_160k_official", seed, 1))
            for seed in cfg["model_seeds"]
        }
        save_json(out / "thresholds.json", {
            "cells": thresholds,
            "level1_cal": level1_thresholds,
            "grid_level1_cal": grid_level1_thresholds,
            "rule_policy": {
                "level_1": ["own_cal"],
                "level_ge_2": ["own_cal", "level1_cal"],
                "level1_cal_source": "same readout at Level 1",
            },
            "family_alpha": eval_cfg["family_alpha"],
            "family_size": eval_cfg["family_size"],
            "per_claim_alpha": alpha,
            "diagnostic_family_sizes": eval_cfg["diagnostic_family_sizes"],
            "audit_min_r2": eval_cfg["audit_min_r2"],
            "cal_gap_quantile": eval_cfg["cal_gap_quantile"],
            "minimum_coverage": eval_cfg["minimum_coverage"],
            "maximum_selective_error": eval_cfg["maximum_selective_error"],
            "lags": eval_cfg["lags"],
            "frozen_before_test_encoding": True,
        })

        # ---------------- Stage (b): sealed TEST prediction ----------------- #
        with h5py.File(ROOT / cfg["data"]["h5"], "r") as data:
            for seed in cfg["model_seeds"]:
                logging.info("Model seed %s: encoding sealed TEST pixels", seed)
                model = load_backbone(cfg, seed, args.device)
                checkpoint_sha = setup["checkpoints"][f"{cfg['model_tag']}/seed{seed}"]["sha256"]
                test_frames = np.asarray(indices["test"]).ravel()
                test_features, test_grid = encode_frames_with_grid(
                    model, data, test_frames, args.device, cfg["encode_batch_size"], needs_grid)
                if cache_ok:
                    stacked = save_or_reuse_features(cfg, seed, "test", test_features,
                                                     {"test": int(test_frames.size)}, checkpoint_sha)
                    test_features = [stacked[level] for level in range(stacked.shape[0])]
                for level in cfg["levels"]:
                    for readout in cfg["_level_readouts"][level]:
                        cell = cell_name(readout, seed, level)
                        kind, feature_kind, artifact = probes[cell]
                        if kind == "8k":
                            predict_fn = predict_probe_8k
                        elif kind == "official":
                            predict_fn = predict_probe_160k_official
                        else:
                            predict_fn = predict_probe_ridge
                        features = test_features[level - 1] if feature_kind == "cls" else test_grid
                        if features is None:
                            raise ValueError(f"Missing {feature_kind} test features for {cell}")
                        prediction = predict_fn(artifact, features, args.device)
                        prediction = prediction.reshape(*indices["test"].shape, 4)
                        np.savez_compressed(out / cell / "test_prediction.npz",
                                            prediction=prediction)
                del model, test_features, test_grid

        predictions = sorted(out.glob("*/test_prediction.npz"))
        if len(predictions) != expected_cells:
            raise ValueError(f"Expected {expected_cells} TEST prediction files, got {len(predictions)}")
        save_json(out / "forecast_receipt.json", {
            "sealed": True,
            "thresholds_sha256": sha256(out / "thresholds.json"),
            "run_script_sha256": sha256(ROOT / "run_v2.py"),
            "metrics_sha256": sha256(ROOT / "metrics.py"),
            "config_sha256": sha256(Path(cfg["_path"])),
            "config_snapshot_sha256": sha256(out / "scripts" / Path(cfg["_path"]).name),
            "config_path": cfg["_path"],
            "data_receipt_sha256": sha256(ROOT / cfg["data"]["receipt"]),
            "prediction_sha256": {path.parent.name: sha256(path) for path in predictions},
        })
        logging.info("Forecast receipt sealed. TEST truth may now be read.")

        # ---------------- Stage (c): unseal and score ----------------------- #
        with h5py.File(ROOT / cfg["data"]["h5"], "r") as data:
            test_truth = states(data, indices["test"])
        np.savez_compressed(out / "test_truth.npz", states=test_truth)

        rows = []
        diagnostics_all = {}
        for level in cfg["levels"]:
            for readout in cfg["_level_readouts"][level]:
                for seed in cfg["model_seeds"]:
                    cell = cell_name(readout, seed, level)
                    prediction = np.load(out / cell / "test_prediction.npz")["prediction"]
                    scores = score_windows(prediction, test_truth, physical_ref_variance,
                                           reference_variance[cell], eval_cfg["lags"])
                    np.savez_compressed(out / cell / "test_scores.npz", **scores)
                    if level == 1:
                        # At Level 1 level1_cal is numerically identical to own_cal,
                        # so it is not a separate Bonferroni claim.
                        rules = {"own_cal": thresholds[cell]}
                    else:
                        rules = {
                            "own_cal": thresholds[cell],
                            "level1_cal": level1_thresholds[f"{readout}/seed{seed}"],
                        }
                    diagnostics = {
                        "cell": cell,
                        "readout": readout,
                        "seed": seed,
                        "level": level,
                        "audit_r2": audit_records[cell]["r2"],
                        "test_r2": dict(zip(AXES, r2_per_axis(prediction, test_truth).tolist())),
                        "predicted_ref_variance": reference_variance[cell].tolist(),
                        "physical_ref_variance": physical_ref_variance.tolist(),
                        "predicted_over_physical_variance": (
                            reference_variance[cell] / physical_ref_variance).tolist(),
                        "cross_term_mean_per_axis": scores["cross_term"].mean(axis=0).tolist(),
                        "cross_term_quantiles": _quantiles(scores["cross_term"]),
                        "error_energy_mean_per_axis": scores["error_energy"].mean(axis=0).tolist(),
                        "error_energy_quantiles": _quantiles(scores["error_energy"]),
                        "window_groups": {},
                        "rules": {},
                        "rule_policy": ("own_cal only at Level 1" if level == 1
                                        else "own_cal and level1_cal at Level >= 2"),
                        "level1_cal_equals_own_cal": level == 1,
                    }
                    if level >= 2 and readout.startswith("cls_"):
                        auxiliary_threshold = grid_level1_thresholds.get(f"seed{seed}")
                        diagnostics["auxiliary_grid_level1_cal"] = auxiliary_threshold
                        if auxiliary_threshold is not None:
                            auxiliary = risk_with(alpha, scores["gap"], scores["wrong"],
                                                  auxiliary_threshold, eval_cfg)
                            diagnostics["auxiliary_grid_level1_rule"] = {
                                "threshold": auxiliary_threshold,
                                "accepted": auxiliary["accepted"],
                                "errors": auxiliary["errors"],
                                "coverage": auxiliary["coverage"],
                                "selective_error": auxiliary["selective_error"],
                                "error_upper": auxiliary["error_upper"],
                            }
                    for rule, threshold in rules.items():
                        result = risk_with(alpha, scores["gap"], scores["wrong"], threshold, eval_cfg)
                        row = {
                            "cell": cell,
                            "readout": readout,
                            "seed": seed,
                            "level": level,
                            "rule": rule,
                            "threshold": threshold,
                            "minimum_r2": float(min(audit_records[cell]["r2"].values())),
                            "eligible": audit_records[cell]["eligible"],
                            **result,
                            "family_size": eval_cfg["family_size"],
                            "per_claim_alpha": alpha,
                        }
                        row["status"] = (
                            "RECOVERY_NOT_VERIFIED" if not row["eligible"]
                            else "PASS" if row["passes"] else "CRITERIA_NOT_MET")
                        diagnostic_bounds = {}
                        for size in eval_cfg["diagnostic_family_sizes"]:
                            diag = risk_with(eval_cfg["family_alpha"] / size, scores["gap"],
                                             scores["wrong"], threshold, eval_cfg)
                            diagnostic_bounds[str(size)] = {
                                "per_claim_alpha": eval_cfg["family_alpha"] / size,
                                "error_upper": diag["error_upper"],
                                "passes": diag["passes"],
                            }
                        row["diagnostic_error_upper"] = diagnostic_bounds
                        rows.append(row)
                        diagnostics["rules"][rule] = {
                            "threshold": threshold,
                            "full_accuracy": float(1 - scores["wrong"].mean()),
                            "accepted": result["accepted"],
                            "errors": result["errors"],
                            "coverage": result["coverage"],
                            "selective_error": result["selective_error"],
                            "error_upper": result["error_upper"],
                            "status": row["status"],
                            "diagnostic_error_upper": diagnostic_bounds,
                        }
                        diagnostics["window_groups"][rule] = group_diagnostics(
                            scores, threshold, alpha, eval_cfg)
                    save_json(out / cell / "diagnostics.json", diagnostics)
                    diagnostics_all[cell] = diagnostics

        save_json(out / "diagnostics_all.json", diagnostics_all)
        save_json(out / "summary.json", {
            "experiment": cfg["experiment"],
            "axes": AXES,
            "family_alpha": eval_cfg["family_alpha"],
            "family_size": eval_cfg["family_size"],
            "per_claim_alpha": alpha,
            "diagnostic_family_sizes": eval_cfg["diagnostic_family_sizes"],
            "rows": rows,
        })
        with (out / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        report = [
            "# Wave-1 FourRoom H-JEPA / LeWM extension", "",
            f"Experiment: `{cfg['experiment']}`. Family size {eval_cfg['family_size']}, "
            f"per-claim alpha {alpha:.6g}.", "",
            "| Cell | Rule | Min R2 | Coverage | Errors/Accepted | Error upper | Status |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            report.append(
                f"| {row['cell']} | {row['rule']} | {row['minimum_r2']:.3f} | "
                f"{row['coverage']:.1%} | {row['errors']}/{row['accepted']} | "
                f"{row['error_upper']:.2%} | {row['status']} |")
        report += [
            "",
            "RECOVERY_NOT_VERIFIED: this readout did not recover all four factors at AUDIT; "
            "it does not prove the representation lost the information.",
            "CRITERIA_NOT_MET: recovery passed but this frozen threshold missed the coverage/error criteria.",
            "PASS: recovery passed, coverage >= 35%, and the Bonferroni-corrected one-sided "
            "Clopper-Pearson error bound is <= 1%.",
            "Coverage is an observed test fraction, not a confidence lower bound.",
            "Level-1 cells report own_cal only: level1_cal is numerically identical there.",
            "grid4x4_* readouts operate on the projected 4x4 patch grid and exist at Level 1 only.",
            "The TEST truth was read only after forecast_receipt.json was sealed.",
        ]
        (out / "RESULTS.md").write_text("\n".join(report) + "\n")
        save_json(out / "status.json", {"status": "COMPLETE"})
        logging.info("Complete: %s", out)
    except BaseException as exc:
        save_json(out / "status.json", {"status": "FAILED", "error": repr(exc)})
        logging.exception("Stopped; keep this result folder for diagnosis")
        raise


if __name__ == "__main__":
    main()
