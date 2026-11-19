"""Two-sprite core for Wave-2 (single-GPU, CPU-smokeable).

Implements a stationary/near-stationary AR(1) four-factor process, a 32x32
two-Gaussian-sprite renderer, the SiLU CNN encoder, SIGReg, LeJEPA-style
pretraining (symmetric or predictor), the 8->128->128->4 probe, and the
orthogonal REF variance decomposition used to classify recovery regimes.

The Pre-Test Forecast Lock is implemented in run_cell():
  thresholds.json -> test_prediction.npz -> forecast_receipt.json -> TEST truth.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from metrics import risk, score_windows  # noqa: E402

IMG = 32
LAGS = (4, 16, 32)
FAST_AXES = (2, 3)
ROLES = ("pretrain", "train", "val", "audit", "ref", "cal", "test")
ROLE_SHAPES = {
    "pretrain": (1024, 128),
    "train": (256, 193),
    "val": (64, 193),
    "audit": (128, 193),
    "ref": (256, 193),
    "cal": (1024, 193),
    "test": (2048, 193),
}
SMOKE_SHAPES = {
    "pretrain": (8, 32),
    "train": (6, 33),
    "val": (4, 33),
    "audit": (4, 33),
    "ref": (6, 33),
    "cal": (8, 33),
    "test": (8, 33),
}
RHO_A = (0.964, 0.961, 0.78, 0.76)
RHO_B = (0.78, 0.76, 0.964, 0.961)
MAX_LAG = 10
N_ANCHORS = 128 - MAX_LAG


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def _ar1(n_episodes, length, rho, rng):
    z = np.empty((n_episodes, length, 4), dtype=np.float32)
    rho = np.asarray(rho, dtype=np.float32).reshape(1, 4)
    scale = np.sqrt(1.0 - rho**2)
    z[:, 0] = rng.standard_normal((n_episodes, 4)).astype(np.float32)
    for t in range(1, length):
        z[:, t] = rho * z[:, t - 1] + scale * rng.standard_normal((n_episodes, 4)).astype(np.float32)
    return z


def generate_splits(data_seed, shapes=None, mixture=None):
    """Generate independent AR(1) episodes per split.

    mixture=None: stationary rho=RHO_A.
    mixture={"p_b": 0.2, "rho_a": RHO_A, "rho_b": RHO_B}: per-episode mode draw.
    """
    shapes = shapes or ROLE_SHAPES
    splits, modes = {}, {}
    for index, role in enumerate(ROLES):
        n_episodes, length = shapes[role]
        rng = np.random.default_rng([int(data_seed), index])
        if mixture is None:
            splits[role] = _ar1(n_episodes, length, RHO_A, rng)
            modes[role] = np.zeros(n_episodes, dtype=np.int8)
        else:
            mode = (rng.random(n_episodes) < float(mixture["p_b"])).astype(np.int8)
            rho = np.where(mode[:, None] == 1, np.asarray(mixture["rho_b"], np.float32),
                           np.asarray(mixture["rho_a"], np.float32))
            z = np.empty((n_episodes, length, 4), dtype=np.float32)
            z[:, 0] = rng.standard_normal((n_episodes, 4)).astype(np.float32)
            for t in range(1, length):
                z[:, t] = (rho * z[:, t - 1]
                           + np.sqrt(1.0 - rho**2)
                           * rng.standard_normal((n_episodes, 4)).astype(np.float32))
            splits[role] = z
            modes[role] = mode
    return splits, modes


def render_states(states, rng=None):
    """Render float32 NCHW 32x32 images from [B,4] states."""
    states = np.asarray(states, dtype=np.float32)
    b = states.shape[0]
    centers = 16.0 + 11.0 * np.tanh(0.45 * states.reshape(b, 2, 2))
    ys, xs = np.mgrid[0:IMG, 0:IMG]
    x = (xs / (IMG - 1)).astype(np.float32)
    y = (ys / (IMG - 1)).astype(np.float32)
    sprites = np.empty((b, 2, IMG, IMG), dtype=np.float32)
    for k in range(2):
        dx = xs[None, :, :] - centers[:, k, 0][:, None, None]
        dy = ys[None, :, :] - centers[:, k, 1][:, None, None]
        sprites[:, k] = np.exp(-(dx * dx + dy * dy) / (2.0 * 2.2**2))
    red = 0.045 + 0.015 * (x * y) + 0.86 * sprites[:, 0]
    blue = 0.050 + 0.015 * ((1.0 - x) * y) + 0.86 * sprites[:, 1]
    green = (0.065 + 0.020 * np.sin(np.pi * (x + y))
             + 0.035 * (sprites[:, 0] + sprites[:, 1]))
    image = np.stack([red, green, blue], axis=1).astype(np.float32)
    if rng is not None:
        image = image + 0.01 * rng.standard_normal(image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Encoder / SIGReg
# --------------------------------------------------------------------------- #

class SpriteEncoder(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(512, 64), nn.LayerNorm(64), nn.Linear(64, latent_dim),
        )

    def forward(self, images):
        return self.features(images)


class SIGReg(nn.Module):
    """Sketched isotropic Gaussian regularizer (Balestriero & LeCun 2025)."""

    def __init__(self, dim=8, knots=17, n_slices=256, t_max=3.0):
        super().__init__()
        t = torch.linspace(0.0, t_max, knots)
        step = t[1] - t[0] if knots > 1 else 1.0
        weights = torch.full_like(t, float(step))
        weights[0] *= 0.5
        weights[-1] *= 0.5
        weights = weights * torch.exp(-t**2 / 2.0)
        directions = F.normalize(torch.randn(dim, n_slices), dim=0)
        self.register_buffer("t", t)
        self.register_buffer("weights", weights)
        self.register_buffer("directions", directions)

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        projection = flat @ self.directions            # (N, S)
        angle = self.t[:, None, None] * projection[None, :, :]  # (K, N, S)
        cos_mean = torch.cos(angle).mean(dim=1)        # (K, S)
        sin_mean = torch.sin(angle).mean(dim=1)        # (K, S)
        phi = torch.exp(-self.t**2 / 2.0)[:, None]     # (K, 1)
        error = (cos_mean - phi) ** 2 + sin_mean**2    # (K, S)
        return (error * self.weights[:, None]).sum(dim=0).mean() * flat.shape[0]


class Predictor(nn.Module):
    def __init__(self, dim=8, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, z):
        return self.net(z)


# --------------------------------------------------------------------------- #
# Pretraining
# --------------------------------------------------------------------------- #

def pretrain_encoder(pretrain_states, *, mode, lag, w_sig, seed, device,
                     steps=6000, batch=256, lr=1e-3, wd=1e-4, log_every=500,
                     rng=None):
    """mode='symmetric_lejepa' or 'predictor'."""
    if mode not in ("symmetric_lejepa", "predictor"):
        raise ValueError(f"Unknown pretrain mode {mode!r}")
    set_seed(seed)
    rng = rng or np.random.default_rng(seed)
    encoder = SpriteEncoder(8).to(device)
    predictor = Predictor(8).to(device) if mode == "predictor" else None
    sigreg = SIGReg(8).to(device)
    parameters = list(encoder.parameters())
    if predictor is not None:
        parameters += list(predictor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=wd)
    n_episodes, seq_len, _ = pretrain_states.shape
    if seq_len < MAX_LAG + 1:
        raise ValueError("pretrain sequences are too short for the shared anchor set")
    anchors = seq_len - MAX_LAG
    trace = []
    encoder.train()
    for step in range(1, steps + 1):
        episode = rng.integers(0, n_episodes, batch)
        anchor = rng.integers(0, anchors, batch)
        current_states = pretrain_states[episode, anchor]
        target_states = pretrain_states[episode, anchor + lag]
        current = torch.from_numpy(render_states(current_states)).to(device)
        target = torch.from_numpy(render_states(target_states)).to(device)
        z_current = encoder(current)
        if mode == "symmetric_lejepa":
            # Symmetric LeJEPA: both views are encoded with gradients.
            z_target = encoder(target)
            matching = F.mse_loss(z_current, z_target)
            latents = torch.cat([z_current, z_target], dim=0)
            loss = (1.0 - w_sig) * matching + w_sig * sigreg(latents)
        else:
            with torch.no_grad():
                z_target = encoder(target)
            predicted = predictor(z_current)
            matching = F.mse_loss(predicted, z_target.detach())
            loss = matching + w_sig * sigreg(z_current.unsqueeze(0))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite pretraining loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            trace.append({"step": step, "loss": float(loss.detach()),
                          "matching": float(matching.detach())})
            logging.info("pretrain %s lag=%s w_sig=%s step %s/%s loss=%.6f",
                         mode, lag, w_sig, step, steps, float(loss.detach()))
    encoder.eval()
    if predictor is not None:
        predictor.eval()
    return encoder, predictor, trace


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #

class Probe(nn.Module):
    def __init__(self, input_dim=8, hidden=128, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


@torch.inference_mode()
def encode_states(encoder, states, device, batch=512):
    states = np.asarray(states, dtype=np.float32)
    output = []
    for start in range(0, len(states), batch):
        image = torch.from_numpy(render_states(states[start:start + batch])).to(device)
        output.append(encoder(image).cpu().numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


def fit_probe(train_x, train_y, val_x, val_y, *, seed, device,
              steps=3000, batch=256, lr=1e-3, wd=1e-4):
    set_seed(seed)
    x_mean, x_std = train_x.mean(0), np.maximum(train_x.std(0), 1e-6)
    y_mean, y_std = train_y.mean(0), train_y.std(0)
    if np.any(y_std <= 0):
        raise ValueError("A probe target has zero variance")
    x = torch.as_tensor((train_x - x_mean) / x_std, device=device)
    y = torch.as_tensor((train_y - y_mean) / y_std, dtype=torch.float32, device=device)
    vx = torch.as_tensor((val_x - x_mean) / x_std, device=device)
    vy = torch.as_tensor((val_y - y_mean) / y_std, dtype=torch.float32, device=device)
    probe = Probe(x.shape[1], 128, 4).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    best_loss, best, trace = float("inf"), None, []
    for step in range(1, steps + 1):
        ids = torch.randint(len(x), (batch,), device=device)
        loss = (probe(x[ids]) - y[ids]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite probe loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            with torch.no_grad():
                val_loss = (probe(vx) - vy).square().mean().item()
            trace.append({"step": step, "train_loss": float(loss.detach()),
                          "val_loss": val_loss})
            if val_loss < best_loss:
                best_loss = val_loss
                best = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
    artifact = {
        "state_dict": best,
        "input_dim": int(x.shape[1]),
        "x_mean": torch.from_numpy(x_mean),
        "x_std": torch.from_numpy(x_std),
        "y_mean": torch.from_numpy(y_mean),
        "y_std": torch.from_numpy(y_std),
        "validation_loss": best_loss,
        "trace": trace,
    }
    return artifact


@torch.inference_mode()
def predict_probe(artifact, features, device):
    probe = Probe(artifact["input_dim"], 128, 4).to(device)
    probe.load_state_dict(artifact["state_dict"])
    probe.eval()
    mean, std = artifact["x_mean"].numpy(), artifact["x_std"].numpy()
    output = []
    for chunk in range(0, len(features), 16384):
        x = torch.as_tensor((features[chunk:chunk + 16384] - mean) / std, device=device)
        output.append(probe(x).cpu().numpy())
    return np.concatenate(output) * artifact["y_std"].numpy() + artifact["y_mean"].numpy()


# --------------------------------------------------------------------------- #
# REF variance decomposition
# --------------------------------------------------------------------------- #

def _monomials_slow(z_slow):
    z0, z1 = z_slow[:, 0], z_slow[:, 1]
    return np.stack([
        z0**2, z0 * z1, z1**2,
        z0**3, z0**2 * z1, z0 * z1**2, z1**3,
        z0**4, z0**3 * z1, z0**2 * z1**2, z0 * z1**3, z1**4,
    ], axis=1)


def _r2(y, design):
    design = np.column_stack([np.ones(len(design)), design])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    projection = design @ beta
    variance = float(np.var(y))
    return float(np.var(projection) / variance) if variance > 0 else 0.0, projection


def _residualize(design, controls):
    controls = np.column_stack([np.ones(len(controls)), controls])
    beta, *_ = np.linalg.lstsq(controls, design, rcond=None)
    return design - controls @ beta


def variance_decomposition(pred_ref, true_ref):
    """Orthogonal decomposition of each fast-axis prediction on REF."""
    pred = np.asarray(pred_ref, dtype=np.float64).reshape(-1, 4)
    true = np.asarray(true_ref, dtype=np.float64).reshape(-1, 4)
    fast = true[:, list(FAST_AXES)]
    slow = true[:, [0, 1]]
    harmonic = _monomials_slow(slow)
    # Standardize features to keep lstsq well conditioned.
    def std(mat):
        mu = mat.mean(0)
        sd = np.maximum(mat.std(0), 1e-8)
        return (mat - mu) / sd
    fast_s, slow_s, harmonic_s = std(fast), std(slow), std(harmonic)
    out = {"axes": {}}
    for j in FAST_AXES:
        y = pred[:, j] - pred[:, j].mean()
        slow_res = _residualize(slow_s, fast_s)
        harmonic_res = _residualize(harmonic_s, np.column_stack([fast_s, slow_s]))
        fast_r2, proj_fast = _r2(y, fast_s)
        slow_r2, proj_slow = _r2(y, slow_res)
        harmonic_r2, proj_harmonic = _r2(y, harmonic_res)
        residual = y - proj_fast - proj_slow - proj_harmonic
        residual_frac = max(0.0, 1.0 - fast_r2 - slow_r2 - harmonic_r2)
        episodes = residual.reshape(pred_ref.shape[0], pred_ref.shape[1])
        correlations = []
        for row in episodes:
            if len(row) > 2 and np.std(row[:-1]) > 1e-12 and np.std(row[1:]) > 1e-12:
                correlations.append(float(np.corrcoef(row[:-1], row[1:])[0, 1]))
        out["axes"][str(j)] = {
            "pred_over_phys_ref_var": float(np.var(pred[:, j]) / np.var(true[:, j])),
            "true_fast_frac": fast_r2,
            "linear_slow_frac": slow_r2,
            "harmonic_slow_frac": harmonic_r2,
            "residual_frac": residual_frac,
            "mean_score_shift": float((pred[:, j] - true[:, j]).mean()
                                      / max(true[:, j].std(), 1e-12)),
            "residual_lag1_autocorr": float(np.nanmean(correlations)) if correlations else None,
        }
    for key in ("pred_over_phys_ref_var", "true_fast_frac", "linear_slow_frac",
                "harmonic_slow_frac", "residual_frac", "mean_score_shift",
                "residual_lag1_autocorr"):
        values = [out["axes"][str(j)][key] for j in FAST_AXES]
        values = [v for v in values if v is not None]
        out[f"mean_{key}"] = float(np.mean(values)) if values else None
    return out


def classify_regime(decomposition):
    fast = decomposition["mean_true_fast_frac"]
    linear = decomposition["mean_linear_slow_frac"]
    harmonic = decomposition["mean_harmonic_slow_frac"]
    if fast >= 0.5:
        return "Normal Recovery"
    if linear >= 0.5 and linear >= harmonic:
        return "Regime II-A"
    if harmonic >= 0.5:
        return "Regime II-B"
    return "Mixed"


# --------------------------------------------------------------------------- #
# Pre-Test Forecast Lock pipeline
# --------------------------------------------------------------------------- #

def _sample_frames(states, per_episode, seed):
    rng = np.random.default_rng(seed)
    indices = np.stack([
        np.sort(rng.choice(states.shape[1], per_episode, replace=False))
        for _ in range(states.shape[0])
    ])
    targets = np.stack([states[i, indices[i]] for i in range(states.shape[0])])
    return targets.reshape(-1, 4)


def _group_stats(scores, threshold):
    gap, wrong = scores["gap"], scores["wrong"]
    regular = (scores["true_top2"][:, 0] == 0) & (scores["true_top2"][:, 1] == 1)
    accept = gap >= threshold
    output = {}
    for name, mask in (("regular_agent_slow", regular), ("local_flip", ~regular)):
        count = int(mask.sum())
        accepted = int((accept & mask).sum())
        errors = int((wrong & accept & mask).sum())
        output[name] = {
            "windows": count,
            "full_accuracy": float(1 - wrong[mask].mean()) if count else None,
            "accepted": accepted,
            "errors": errors,
            "selective_error": errors / accepted if accepted else None,
        }
    return output


def run_cell(encoder, data, *, out_dir, cell, seed, alpha, device,
             train_per_episode=32, val_per_episode=32,
             probe_steps=3000, probe_batch=256, lags=LAGS,
             audit_min_r2=0.85, minimum_coverage=0.35,
             maximum_selective_error=0.01, extra_meta=None):
    """Fit probe, freeze threshold, seal TEST predictions, then score TEST truth."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_states = _sample_frames(data["train"], train_per_episode, seed + 11)
    val_states = _sample_frames(data["val"], val_per_episode, seed + 17)
    train_x = encode_states(encoder, train_states, device)
    val_x = encode_states(encoder, val_states, device)
    probe = fit_probe(train_x, train_states, val_x, val_states,
                      seed=seed, device=device, steps=probe_steps, batch=probe_batch)
    torch.save(probe, out_dir / "probe.pt")
    save_json(out_dir / "probe.json", {
        "seed": seed, "train_frames": int(len(train_x)), "val_frames": int(len(val_x)),
        "validation_loss": probe["validation_loss"], "trace": probe["trace"],
    })

    latents = {role: encode_states(encoder, data[role].reshape(-1, 4), device)
               for role in ("audit", "ref", "cal", "test")}
    predictions = {role: predict_probe(probe, latents[role], device).reshape(
        data[role].shape[0], data[role].shape[1], 4)
        for role in ("audit", "ref", "cal", "test")}

    audit_truth = data["audit"]
    audit_r2 = 1 - np.mean((predictions["audit"] - audit_truth) ** 2, axis=(0, 1)) \
        / audit_truth.reshape(-1, 4).var(0)
    audit = {"r2": dict(zip([f"z{i}" for i in range(4)], audit_r2.tolist())),
             "minimum_r2": float(np.min(audit_r2)),
             "eligible": bool(np.min(audit_r2) >= audit_min_r2)}
    save_json(out_dir / "audit.json", audit)
    np.savez_compressed(out_dir / "audit.npz",
                        prediction=predictions["audit"], truth=audit_truth)

    true_ref = data["ref"]
    physical_var = true_ref.reshape(-1, 4).var(0)
    predicted_var = predictions["ref"].reshape(-1, 4).var(0)
    np.savez_compressed(out_dir / "reference.npz",
                        prediction=predictions["ref"], variance=predicted_var)

    cal_prediction = predictions["cal"]
    cal_scores = score_windows(cal_prediction, cal_prediction, predicted_var,
                               predicted_var, lags)
    threshold = float(np.quantile(cal_scores["gap"], 0.5))
    save_json(out_dir / "thresholds.json",
              {"cell": cell, "own_cal": threshold, "family_alpha": 0.05,
               "per_claim_alpha": alpha, "frozen_before_test_encoding": True})
    np.savez_compressed(out_dir / "calibration.npz",
                        prediction=cal_prediction, gap=cal_scores["gap"],
                        scores=cal_scores["predicted_scores"])

    # Stage (b): every TEST prediction is written before the truth is scored.
    np.savez_compressed(out_dir / "test_prediction.npz",
                        prediction=predictions["test"])
    save_json(out_dir / "forecast_receipt.json", {
        "sealed": True,
        "cell": cell,
        "cell_thresholds_sha256": sha256(out_dir / "thresholds.json"),
        "probe_sha256": sha256(out_dir / "probe.pt"),
        "test_prediction_sha256": sha256(out_dir / "test_prediction.npz"),
    })

    # Stage (c): unseal TEST truth.
    test_truth = data["test"]
    test_scores = score_windows(predictions["test"], test_truth,
                                physical_var, predicted_var, lags)
    np.savez_compressed(out_dir / "test_scores.npz", **test_scores)
    result = risk(test_scores["gap"], test_scores["wrong"], threshold, alpha,
                  minimum_coverage, maximum_selective_error)
    test_r2 = 1 - np.mean((predictions["test"] - test_truth) ** 2, axis=(0, 1)) \
        / test_truth.reshape(-1, 4).var(0)
    decomposition = variance_decomposition(predictions["ref"], true_ref)
    diagnostics = {
        "cell": cell,
        "audit_r2": audit["r2"],
        "test_r2": dict(zip([f"z{i}" for i in range(4)], test_r2.tolist())),
        "predicted_ref_variance": predicted_var.tolist(),
        "physical_ref_variance": physical_var.tolist(),
        "predicted_over_physical_variance": (predicted_var / physical_var).tolist(),
        "decomposition": decomposition,
        "regime": classify_regime(decomposition),
        "window_groups": _group_stats(test_scores, threshold),
        "threshold": threshold,
        "risk": result,
        "meta": extra_meta or {},
    }
    save_json(out_dir / "diagnostics.json", diagnostics)
    row = {
        "cell": cell,
        "seed": seed,
        "threshold": threshold,
        "minimum_r2": audit["minimum_r2"],
        "eligible": audit["eligible"],
        **result,
        "per_claim_alpha": alpha,
        "status": ("RECOVERY_NOT_VERIFIED" if not audit["eligible"]
                   else "PASS" if result["passes"] else "CRITERIA_NOT_MET"),
    }
    return row, diagnostics, {
        "gap": test_scores["gap"],
        "wrong": test_scores["wrong"],
        "true_top2": test_scores["true_top2"],
        "predicted_top2": test_scores["predicted_top2"],
        "true_gap": test_scores["true_gap"],
        "modes": data.get("_test_modes"),
    }
