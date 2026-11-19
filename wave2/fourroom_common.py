"""Guard-free FourRoom helpers for the single-GPU Wave-2 AutoDL suite.

This module deliberately does not import run_v2.py (whose entry guard only
permits GPUs 6/7). It contains the minimal audited Wave-1 machinery needed by
wave2/s1_fourroom_dim192.py:

  * official checkpoint loading and dense CLS + 4x4 spatial-grid extraction
  * official 60-epoch two-head probes and closed-form Ridge readouts
  * HDF5 split/state reading and per-axis R^2

Scoring and risk certification must still use the untouched root metrics.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import h5py
import hdf5plugin  # noqa: F401  (registers the collector compression filter)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/H-JEPA"
AXES = ["agent_x", "agent_y", "distractor_x", "distractor_y"]
NONTEST_ROLES = ("train", "val", "audit", "ref", "cal")
MODEL_TAGS = {"fourroom_lewm": 1, "fourroom_hjepa_l2": 2, "fourroom_hjepa_l4": 4}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def upstream_paths():
    for path in (str(SOURCE / "h_jepa"), str(SOURCE)):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_splits(path):
    raw = json.loads(Path(path).read_text())
    roles = {role: np.asarray(value["frame_indices"], dtype=np.int64)
             for role, value in raw["roles"].items()}
    small = {name: np.asarray(value["frame_indices"], dtype=np.int64)
             for name, value in raw.get("small", {}).items()}
    return raw, roles, small


def states(data, indices):
    values = []
    for row in np.asarray(indices):
        start, stop = int(row[0]), int(row[-1]) + 1
        values.append(np.concatenate(
            [data["xy"][start:stop], data["distractor0_xy"][start:stop]], axis=1)[row - start])
    return np.stack(values).astype(np.float64)


def load_backbone(tag, seed, device):
    if tag not in MODEL_TAGS:
        raise ValueError(f"Unsupported model tag {tag!r}")
    upstream_paths()
    path = ROOT / f"assets/weights/fourroom/{tag}/seed{seed}/{tag}_object.ckpt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint {path}; run wave2/setup_autodl.sh")
    model = torch.load(path, map_location="cpu", weights_only=False).eval().to(device)
    model.requires_grad_(False)
    if model.num_levels != MODEL_TAGS[tag]:
        raise ValueError(f"{tag} has {model.num_levels} levels")
    for level in range(1, model.num_levels + 1):
        jepa = model.get_level(level)
        if int(getattr(jepa, "temporal_window_size", 1)) != 1:
            raise ValueError(f"{tag} level {level} window_size != 1")
        if int(model.embed_dims[level]) != 192:
            raise ValueError(f"{tag} level {level} embed dim != 192")
    return model


def load_normalizer(tag, seed):
    path = ROOT / f"assets/weights/fourroom/{tag}/seed{seed}/normalizer.pt"
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


def read_pixels(dataset, batch):
    batch = np.asarray(batch, dtype=np.int64)
    if batch.size == 0 or (batch.size > 1 and not np.all(np.diff(batch) > 0)):
        raise ValueError("Pixel frame batch must be nonempty and strictly increasing")
    cuts = np.flatnonzero(np.diff(batch) != 1)
    starts = np.r_[0, cuts + 1]
    stops = np.r_[cuts + 1, batch.size]
    parts = [dataset[int(batch[s]):int(batch[e - 1]) + 1] for s, e in zip(starts, stops)]
    return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)


def _level1_cls_and_grid(model, normalized_pixels):
    encoder = model.get_level(1).encoder
    if not (hasattr(encoder, "encoder") and hasattr(encoder, "projector")):
        raise ValueError("Level-1 encoder is not a ProjectedEncoder")
    hidden = encoder.encoder(normalized_pixels,
                             interpolate_pos_encoding=True).last_hidden_state
    if hidden.shape[1] != 257 or hidden.shape[-1] != 192:
        raise ValueError(f"Expected CLS + 256 patch tokens of width 192, got {tuple(hidden.shape)}")
    cls = encoder.projector(hidden[:, 0])
    patches = hidden[:, 1:]
    projected = encoder.projector(
        patches.reshape(-1, patches.shape[-1])).reshape(patches.shape)
    grid = projected.reshape(-1, 16, 16, projected.shape[-1])
    grid = grid.reshape(-1, 4, 4, 4, 4, projected.shape[-1]).mean(dim=(2, 4))
    return cls, grid.reshape(grid.shape[0], -1)


@torch.inference_mode()
def encode_frames_with_grid(model, data, frame_indices, device, batch_size, want_grid=True):
    frames = np.asarray(frame_indices, dtype=np.int64).ravel()
    cls_result = [[] for _ in range(model.num_levels)]
    grid_result = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    started = time.time()
    for start in range(0, len(frames), batch_size):
        batch = frames[start:start + batch_size]
        pixels = torch.from_numpy(np.ascontiguousarray(read_pixels(data["pixels"], batch)))
        pixels = pixels.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255
        cls, grid = _level1_cls_and_grid(model, (pixels - mean) / std)
        embedded = cls[:, None, :]
        cls_result[0].append(cls.cpu().numpy())
        if want_grid:
            grid_result.append(grid.cpu().numpy())
        for level in range(2, model.num_levels + 1):
            key = f"embed_{level - 1}"
            embedded = model.get_level(level).encode({key: embedded}, key=key)["embed_0"]
            cls_result[level - 1].append(embedded[:, 0].cpu().numpy())
        if start and start % (batch_size * 200) == 0:
            logging.info("Encoded %s / %s frames (%.0f frames/s)", start, len(frames),
                         start / max(time.time() - started, 1e-9))
    output = [np.concatenate(part).astype(np.float32, copy=False) for part in cls_result]
    grid_output = np.concatenate(grid_result).astype(np.float32, copy=False) if want_grid else None
    if any(not np.isfinite(level).all() for level in output):
        raise ValueError("Nonfinite backbone features")
    if grid_output is not None and not np.isfinite(grid_output).all():
        raise ValueError("Nonfinite grid features")
    return output, grid_output


def encode_roles_with_grid(model, data, indices, roles, device, batch_size, want_grid=True):
    cls_per_role, grid_per_role, lengths = {}, {}, {}
    for role in roles:
        frames = np.asarray(indices[role], dtype=np.int64).ravel()
        if frames.size and not np.all(np.diff(frames) > 0):
            raise ValueError(f"Frame indices for role {role} are not strictly increasing")
        cls, grid = encode_frames_with_grid(model, data, frames, device, batch_size, want_grid)
        cls_per_role[role] = cls
        if want_grid:
            grid_per_role[role] = grid
        lengths[role] = int(frames.size)
    num_levels = len(cls_per_role[roles[0]])
    sliced = {role: [cls_per_role[role][level] for level in range(num_levels)] for role in roles}
    return sliced, (grid_per_role if want_grid else None), lengths


def make_official_mlp(input_dim, hidden, output_dim, device):
    upstream_paths()
    from models.module import MLP  # noqa: PLC0415
    return MLP(input_dim=input_dim, hidden_dim=hidden, output_dim=output_dim).to(device)


def fit_probe_official(train_x, train_y, val_x, val_y, normalizer, seed, pcfg, device, folder):
    """Audited official 60-epoch two-head probe (raw input, normalizer targets)."""
    x_train = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    x_val = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    columns = list(pcfg["target_columns"])
    means, stds, targets, val_targets, heads = {}, {}, {}, {}, {}
    torch.manual_seed(seed)
    for name in columns:
        lo, hi = pcfg["target_columns"][name]
        mean, std = normalizer[name]
        means[name], stds[name] = mean, std
        targets[name] = torch.as_tensor((train_y[:, lo:hi] - mean) / std,
                                        dtype=torch.float32, device=device)
        val_targets[name] = torch.as_tensor((val_y[:, lo:hi] - mean) / std,
                                            dtype=torch.float32, device=device)
        heads[name] = make_official_mlp(x_train.shape[1], pcfg["hidden"], hi - lo, device)
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
    artifact = {
        "format": "wave2_official_probe_v1",
        "input_dim": int(x_train.shape[1]),
        "hidden": int(pcfg["hidden"]),
        "columns": {},
        "seed": seed,
        "epochs": int(pcfg["epochs"]),
        "batch_size": batch_size,
        "learning_rate": float(pcfg["learning_rate"]),
        "weight_decay": float(pcfg["weight_decay"]),
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
    torch.save(artifact, folder / "probe_official.pt")
    save_json(folder / "probe_official.json", {
        "seed": seed, "trace": trace, "train_frames": int(x_train.shape[0]),
        "val_frames": int(x_val.shape[0]), "final_val_losses": trace[-1],
    })
    return artifact


@torch.inference_mode()
def predict_probe_official(artifact, features, device):
    output = []
    for name, column in artifact["columns"].items():
        head = make_official_mlp(artifact["input_dim"], artifact["hidden"],
                                 column["target_slice"][1] - column["target_slice"][0], device)
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
        "format": "wave2_probe_ridge_v1",
        "input_dim": int(train_x.shape[1]),
        "x_mean": x_mean.detach().cpu(),
        "x_std": x_std.detach().cpu(),
        "ridge_lambdas": lambdas,
        "columns": {},
        "selection": "lambda_by_val_mse",
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
    torch.save(artifact, folder / "probe_ridge.pt")
    save_json(folder / "probe_ridge.json", trace)
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


def r2_per_axis(prediction, truth):
    variance = truth.reshape(-1, truth.shape[-1]).var(0)
    if np.any(variance <= 0):
        raise ValueError("A factor has zero variance")
    return 1 - np.mean((prediction - truth) ** 2,
                       axis=tuple(range(prediction.ndim - 1))) / variance


def fit_probe_steps(train_x, train_y, val_x, val_y, normalizer, seed, pcfg, device,
                    folder=None):
    """Official two-head MLP trained for a fixed number of gradient steps.

    Used only by the S1-B sample-budget curve. Raw input, normalizer targets,
    AdamW, validation-MSE selection every 100 steps.
    """
    x_train = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    x_val = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    columns = list(pcfg["target_columns"])
    means, stds, targets, val_targets, heads = {}, {}, {}, {}, {}
    torch.manual_seed(seed)
    for name in columns:
        lo, hi = pcfg["target_columns"][name]
        mean, std = normalizer[name]
        means[name], stds[name] = mean, std
        targets[name] = torch.as_tensor((train_y[:, lo:hi] - mean) / std,
                                        dtype=torch.float32, device=device)
        val_targets[name] = torch.as_tensor((val_y[:, lo:hi] - mean) / std,
                                            dtype=torch.float32, device=device)
        heads[name] = make_official_mlp(x_train.shape[1], pcfg["hidden"], hi - lo, device)
    parameters = [p for head in heads.values() for p in head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=pcfg["learning_rate"],
                                  weight_decay=pcfg["weight_decay"])
    steps = int(pcfg["steps"])
    batch = int(pcfg["batch_size"])
    n = x_train.shape[0]
    best_loss, best_state, trace = float("inf"), None, []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for step in range(1, steps + 1):
        ids = torch.randint(n, (batch,), device=device, generator=generator)
        loss = None
        for name, head in heads.items():
            head.train()
            term = F.mse_loss(head(x_train[ids]), targets[name][ids])
            loss = term if loss is None else loss + term
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            with torch.no_grad():
                val_loss = None
                for name, head in heads.items():
                    head.eval()
                    term = float(F.mse_loss(head(x_val), val_targets[name]))
                    val_loss = term if val_loss is None else val_loss + term
            trace.append({"step": step, "train_loss": float(loss.detach()),
                          "val_loss": val_loss})
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {name: {k: v.detach().cpu().clone()
                                     for k, v in head.state_dict().items()}
                              for name, head in heads.items()}
    artifact = {
        "format": "wave2_budget_probe_v1",
        "input_dim": int(x_train.shape[1]),
        "hidden": int(pcfg["hidden"]),
        "columns": {},
        "seed": seed,
        "steps": steps,
        "batch_size": batch,
        "learning_rate": float(pcfg["learning_rate"]),
        "weight_decay": float(pcfg["weight_decay"]),
        "selection": "best_val_mse",
        "train_frames": int(x_train.shape[0]),
        "val_frames": int(x_val.shape[0]),
    }
    for name in columns:
        lo, hi = pcfg["target_columns"][name]
        artifact["columns"][name] = {
            "state_dict": best_state[name],
            "target_slice": [int(lo), int(hi)],
            "target_mean": torch.from_numpy(means[name]),
            "target_std": torch.from_numpy(stds[name]),
        }
    if folder is not None:
        Path(folder).mkdir(parents=True, exist_ok=True)
        torch.save(artifact, Path(folder) / "probe.pt")
        save_json(Path(folder) / "probe.json",
                  {"seed": seed, "trace": trace, "validation_loss": best_loss,
                   "train_frames": int(x_train.shape[0]),
                   "val_frames": int(x_val.shape[0])})
    return artifact, best_loss, trace
