"""S1: FourRoom Level-1 readout dimensionality (192-D projections).

Reuses the audited Wave-1 backbone/grid/readout machinery without importing
run_v2.py. For every model family (LeWM, H-JEPA L2, H-JEPA L4) x seed 42/43/44
at Level 1 it fits an official 192-D MLP and a 192-D Ridge on:

  grid4x4_pca192  deterministic PCA (float64 covariance eigh) on TRAIN only
  grid4x4_rand192 fixed orthonormal QR random projection

36 confirmatory cells, family_size=36 (alpha=0.05/36), M=9 and M=51 diagnostics.
The global Pre-Test Forecast Lock is enforced across all seeds:
  (a) fit probes/thresholds on nontest roles only;
  (b) encode TEST once per checkpoint and seal every test_prediction.npz;
  (c) only then read TEST truth and score.

Also writes the S1-B sample-budget curve (steps=3000, batch=128) for
fourroom_hjepa_l2 (cls, grid_pca192, grid_3072) and fourroom_lewm (cls).

    python3 wave2/s1_fourroom_dim192.py --device cuda
    python3 wave2/s1_fourroom_dim192.py --self-test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fourroom_common as fc  # noqa: E402

ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from metrics import risk, score_windows  # noqa: E402

DATA = ROOT / "assets/data/fourroom_v2"
OUT = ROOT / "results/wave2_fourroom_dim192"
MODEL_TAGS = ("fourroom_lewm", "fourroom_hjepa_l2", "fourroom_hjepa_l4")
SEEDS = (42, 43, 44)
PCA_DIM = 192
RAND_BASE = 2026111900
LAGS = (4, 16, 32)
BUDGETS = (256, 1024, 4096, 16384, 65536, 160400)
OFFICIAL_CFG = {
    "hidden": 512,
    "epochs": 60,
    "steps": 3000,
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
CELL_SPECS = (
    ("mlp", "official"),
    ("ridge", "ridge"),
)
PROJECTIONS = ("pca192", "rand192")


def pca_fit(train_grid, dim=PCA_DIM, device="cuda"):
    """PCA via float64 covariance eigh on TRAIN frames only."""
    x = torch.as_tensor(train_grid, dtype=torch.float64, device=device)
    mean = x.mean(0)
    centered = x - mean
    covariance = centered.T @ centered / max(x.shape[0] - 1, 1)
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True)
    values = values[order]
    vectors = vectors[:, order]
    top = vectors[:, :dim]
    explained = float(values[:dim].sum() / values.sum())
    return (mean.cpu().numpy().astype(np.float64),
            top.cpu().numpy().astype(np.float64),
            explained,
            values[:dim].cpu().numpy().astype(np.float64))


def random_projection(seed, dim=PCA_DIM):
    rng = np.random.default_rng(RAND_BASE + int(seed))
    matrix = rng.standard_normal((3072, dim))
    q, _ = np.linalg.qr(matrix)
    return q.astype(np.float64)


def apply_projection(grid, mean, basis, device="cuda", chunk=8192):
    output = []
    mean_t = torch.as_tensor(mean, dtype=torch.float64, device=device)
    basis_t = torch.as_tensor(basis, dtype=torch.float64, device=device)
    for start in range(0, len(grid), chunk):
        x = torch.as_tensor(grid[start:start + chunk], dtype=torch.float64, device=device)
        output.append(((x - mean_t) @ basis_t).float().cpu().numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


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


def _residual_lag1(prediction, truth):
    residual = prediction - truth
    correlations = []
    for axis in range(4):
        for row in residual[:, :, axis]:
            if len(row) > 2 and np.std(row[:-1]) > 1e-12 and np.std(row[1:]) > 1e-12:
                correlations.append(float(np.corrcoef(row[:-1], row[1:])[0, 1]))
    return float(np.nanmean(correlations)) if correlations else None


def check_receipt():
    path = DATA / "data_receipt.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run wave2/prepare_data_autodl.py first")
    receipt = json.loads(path.read_text())
    if fc.sha256(DATA / "fourroom.h5") != receipt["h5_sha256"]:
        raise ValueError("fourroom.h5 no longer matches data_receipt.json")
    if fc.sha256(DATA / "splits_v2.json") != receipt["splits_sha256"]:
        raise ValueError("splits_v2.json no longer matches data_receipt.json")
    return receipt


def self_test():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((500, 3072)).astype(np.float32)
    mean, basis, explained, _ = pca_fit(x, dim=PCA_DIM, device="cpu")
    if basis.shape != (3072, PCA_DIM):
        raise AssertionError("PCA basis shape")
    if not np.allclose(basis.T @ basis, np.eye(PCA_DIM), atol=1e-6):
        raise AssertionError("PCA basis is not orthonormal")
    if not 0.0 < explained <= 1.0 + 1e-9:
        raise AssertionError("PCA explained variance out of range")
    q = random_projection(0)
    if q.shape != (3072, PCA_DIM):
        raise AssertionError("QR basis shape")
    if not np.allclose(q.T @ q, np.eye(PCA_DIM), atol=1e-10):
        raise AssertionError("QR basis is not orthonormal")
    projected = apply_projection(x, mean, basis, device="cpu")
    projected_random = apply_projection(x, np.zeros(3072), q, device="cpu")
    if projected.shape != (500, PCA_DIM) or projected_random.shape != (500, PCA_DIM):
        raise AssertionError("projection shape")
    print(f"SELF_TEST_OK PCA_EXPLAINED={explained:.6f}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--limit-models", nargs="*", default=None)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    check_receipt()
    _, splits, _ = fc.load_splits(DATA / "splits_v2.json")
    model_tags = tuple(args.limit_models) if args.limit_models else MODEL_TAGS
    family_size = len(model_tags) * len(SEEDS) * len(PROJECTIONS) * len(CELL_SPECS)
    alpha = 0.05 / family_size
    logging.info("S1 family_size=%s alpha=%s", family_size, alpha)

    with h5py.File(DATA / "fourroom.h5", "r") as data:
        physical_ref = fc.states(data, splits["ref"])
        physical_var = physical_ref.reshape(-1, 4).var(0)
        audit_truth = fc.states(data, splits["audit"])
        thresholds, probes, reference_var, audit_records = {}, {}, {}, {}
        projections, cells_meta, projection_meta = {}, {}, {}
        budget_rows = []
        train_truth = fc.states(data, splits["train"]).reshape(-1, 4)
        val_truth = fc.states(data, splits["val"]).reshape(-1, 4)

        # ---------------- Stage (a): nontest only ------------------------- #
        for tag in model_tags:
            for seed in SEEDS:
                logging.info("S1 stage(a) %s seed %s", tag, seed)
                model = fc.load_backbone(tag, seed, args.device)
                sliced, grid, _ = fc.encode_roles_with_grid(
                    model, data, splits, fc.NONTEST_ROLES, args.device, 256, True)
                del model
                normalizer = fc.load_normalizer(tag, seed)
                cls_l1 = {role: sliced[role][0] for role in fc.NONTEST_ROLES}
                mean, basis, explained, eigenvalues = pca_fit(grid["train"], PCA_DIM, args.device)
                q_basis = random_projection(seed)
                projections[(tag, seed)] = {
                    "pca192": {"mean": mean, "basis": basis, "explained": explained,
                               "top_eigenvalues": eigenvalues.tolist()},
                    "rand192": {"mean": np.zeros(3072, np.float64), "basis": q_basis},
                }
                projection_meta[f"{tag}_seed{seed}"] = {
                    "model": tag, "seed": seed,
                    "pca_explained_variance_ratio": explained,
                    "pca_top_eigenvalues": eigenvalues.tolist(),
                    "random_projection_seed": RAND_BASE + seed,
                }
                fc.save_json(OUT / f"projection_{tag}_seed{seed}.json",
                             projection_meta[f"{tag}_seed{seed}"])
                proj_features = {}
                for projection in PROJECTIONS:
                    info = projections[(tag, seed)][projection]
                    proj_features[projection] = {
                        role: apply_projection(grid[role], info["mean"], info["basis"],
                                               args.device)
                        for role in fc.NONTEST_ROLES}
                for projection in PROJECTIONS:
                    for readout, _kind in CELL_SPECS:
                        cell = f"{tag}_{projection}_{readout}_seed{seed}"
                        folder = OUT / cell
                        folder.mkdir(parents=True, exist_ok=True)
                        train_x = proj_features[projection]["train"]
                        val_x = proj_features[projection]["val"]
                        if readout == "mlp":
                            artifact = fc.fit_probe_official(
                                train_x, train_truth, val_x, val_truth,
                                normalizer, seed, OFFICIAL_CFG, args.device, folder)
                            predict_fn = fc.predict_probe_official
                        else:
                            artifact = fc.fit_probe_ridge(
                                train_x, train_truth, val_x, val_truth,
                                normalizer, RIDGE_CFG, args.device, folder)
                            predict_fn = fc.predict_probe_ridge
                        probes[cell] = (readout, projection, artifact)
                        cells_meta[cell] = {"model": tag, "projection": projection,
                                            "readout": readout, "seed": seed}
                        audit_pred = predict_fn(artifact, proj_features[projection]["audit"],
                                                args.device).reshape(*splits["audit"].shape, 4)
                        r2 = fc.r2_per_axis(audit_pred, audit_truth)
                        audit_records[cell] = {
                            "r2": dict(zip(fc.AXES, r2.tolist())),
                            "minimum_r2": float(np.min(r2)),
                            "eligible": bool(np.min(r2) >= 0.85),
                        }
                        fc.save_json(folder / "audit.json", audit_records[cell])
                        np.savez_compressed(folder / "audit.npz",
                                            prediction=audit_pred, truth=audit_truth)
                        ref_pred = predict_fn(artifact, proj_features[projection]["ref"],
                                              args.device).reshape(*splits["ref"].shape, 4)
                        ref_var = ref_pred.reshape(-1, 4).var(0)
                        reference_var[cell] = ref_var
                        np.savez_compressed(folder / "reference.npz",
                                            prediction=ref_pred, variance=ref_var)
                        cal_pred = predict_fn(artifact, proj_features[projection]["cal"],
                                              args.device).reshape(*splits["cal"].shape, 4)
                        cal_scores = score_windows(cal_pred, cal_pred, ref_var, ref_var, LAGS)
                        thresholds[cell] = float(np.quantile(cal_scores["gap"], 0.5))
                        fc.save_json(folder / "thresholds.json",
                                     {"cell": cell, "own_cal": thresholds[cell],
                                      "family_size": family_size,
                                      "per_claim_alpha": alpha,
                                      "frozen_before_test_encoding": True})
                        np.savez_compressed(folder / "calibration.npz",
                                            prediction=cal_pred, gap=cal_scores["gap"],
                                            scores=cal_scores["predicted_scores"])
                        logging.info("S1 %s audit=%.4f threshold=%.6f", cell,
                                     float(np.min(r2)), thresholds[cell])
                # S1-B budget curve (nontest only)
                if tag in ("fourroom_hjepa_l2", "fourroom_lewm"):
                    budget_readouts = {
                        "cls": (cls_l1["train"], cls_l1["val"], cls_l1["audit"]),
                    }
                    if tag == "fourroom_hjepa_l2":
                        budget_readouts["grid4x4_pca192"] = (
                            proj_features["pca192"]["train"],
                            proj_features["pca192"]["val"],
                            proj_features["pca192"]["audit"])
                        budget_readouts["grid4x4_3072"] = (
                            grid["train"], grid["val"], grid["audit"])
                    for readout_name, (train_features, val_features, audit_features) in \
                            budget_readouts.items():
                        budget_cfg = dict(OFFICIAL_CFG)
                        budget_cfg["steps"] = 3000
                        budget_cfg["batch_size"] = 128
                        rng = np.random.default_rng(RAND_BASE + seed)
                        n_total = len(train_features)
                        for n_train in BUDGETS:
                            n_use = min(n_train, n_total)
                            ids = (np.arange(n_use) if n_use == n_total else
                                   np.sort(rng.choice(n_total, n_use, replace=False)))
                            artifact, val_loss, _ = fc.fit_probe_steps(
                                train_features[ids], train_truth[ids],
                                val_features, val_truth,
                                normalizer, seed, budget_cfg, args.device, None)
                            audit_pred = fc.predict_probe_official(
                                artifact, audit_features, args.device).reshape(
                                    *splits["audit"].shape, 4)
                            r2 = fc.r2_per_axis(audit_pred, audit_truth)
                            budget_rows.append({
                                "model": tag, "seed": seed, "readout": readout_name,
                                "n_train": int(n_use), "audit_min_r2": float(np.min(r2)),
                                "validation_loss": float(val_loss), "steps": 3000,
                            })
                            logging.info("budget %s %s %s n=%s audit=%.4f", tag, readout_name,
                                         seed, n_use, float(np.min(r2)))
                del sliced, grid, proj_features

        # ---------------- Stage (b): seal TEST predictions ----------------- #
        predictions = {}
        for tag in model_tags:
            for seed in SEEDS:
                logging.info("S1 stage(b) %s seed %s", tag, seed)
                model = fc.load_backbone(tag, seed, args.device)
                test_frames = np.asarray(splits["test"]).ravel()
                cls_test, grid_test = fc.encode_frames_with_grid(
                    model, data, test_frames, args.device, 256, True)
                del model
                test_features = {
                    "cls": cls_test[0],
                    "pca192": apply_projection(grid_test, projections[(tag, seed)]["pca192"]["mean"],
                                               projections[(tag, seed)]["pca192"]["basis"], args.device),
                    "rand192": apply_projection(grid_test, projections[(tag, seed)]["rand192"]["mean"],
                                                projections[(tag, seed)]["rand192"]["basis"], args.device),
                }
                for projection in PROJECTIONS:
                    for readout, _ in CELL_SPECS:
                        cell = f"{tag}_{projection}_{readout}_seed{seed}"
                        _readout, _projection, artifact = probes[cell]
                        predict_fn = (fc.predict_probe_official if readout == "mlp"
                                      else fc.predict_probe_ridge)
                        prediction = predict_fn(artifact, test_features[projection],
                                                args.device).reshape(*splits["test"].shape, 4)
                        predictions[cell] = prediction
                        folder = OUT / cell
                        np.savez_compressed(folder / "test_prediction.npz",
                                            prediction=prediction)
                        fc.save_json(folder / "forecast_receipt.json", {
                            "sealed": True, "cell": cell,
                            "thresholds_sha256": fc.sha256(folder / "thresholds.json"),
                            "probe_sha256": fc.sha256(
                                folder / ("probe_official.pt" if readout == "mlp"
                                          else "probe_ridge.pt")),
                            "test_prediction_sha256": fc.sha256(
                                folder / "test_prediction.npz"),
                        })
                del cls_test, grid_test, test_features
        fc.save_json(OUT / "forecast_receipt.json", {
            "sealed_before_test_truth": True,
            "test_prediction_sha256": {
                cell: fc.sha256(OUT / cell / "test_prediction.npz")
                for cell in predictions},
        })
        logging.info("S1 forecast receipts sealed; reading TEST truth now")
        test_truth = fc.states(data, splits["test"])

        # ---------------- Stage (c): score ------------------------------- #
        rows, diagnostics_all, budget_summary = [], {}, {}
        for cell, prediction in predictions.items():
            meta = cells_meta[cell]
            tag, projection, readout, seed = (meta["model"], meta["projection"],
                                              meta["readout"], meta["seed"])
            scores = score_windows(prediction, test_truth, physical_var,
                                   reference_var[cell], LAGS)
            np.savez_compressed(OUT / cell / "test_scores.npz", **scores)
            result = risk(scores["gap"], scores["wrong"], thresholds[cell], alpha, 0.35, 0.01)
            diagnostic_bounds = {}
            for size in (9, 51):
                bound = risk(scores["gap"], scores["wrong"], thresholds[cell],
                             0.05 / size, 0.35, 0.01)
                diagnostic_bounds[str(size)] = {
                    "per_claim_alpha": 0.05 / size,
                    "error_upper": bound["error_upper"],
                    "passes": bound["passes"],
                }
            test_r2 = fc.r2_per_axis(prediction, test_truth)
            diagnostics = {
                "cell": cell,
                "model": tag,
                "seed": seed,
                "projection": projection,
                "readout": readout,
                "audit_r2": audit_records[cell]["r2"],
                "test_r2": dict(zip(fc.AXES, test_r2.tolist())),
                "predicted_ref_variance": reference_var[cell].tolist(),
                "physical_ref_variance": physical_var.tolist(),
                "predicted_over_physical_variance": (
                    reference_var[cell] / physical_var).tolist(),
                "window_groups": _group_stats(scores, thresholds[cell]),
                "test_residual_lag1_autocorr": _residual_lag1(prediction, test_truth),
                "diagnostic_error_upper": diagnostic_bounds,
            }
            fc.save_json(OUT / cell / "diagnostics.json", diagnostics)
            diagnostics_all[cell] = diagnostics
            rows.append({
                "cell": cell, "model": tag, "seed": seed, "projection": projection,
                "readout": readout, "threshold": thresholds[cell],
                "minimum_r2": audit_records[cell]["minimum_r2"],
                "eligible": audit_records[cell]["eligible"],
                **result, "family_size": family_size, "per_claim_alpha": alpha,
                "diagnostic_error_upper": diagnostic_bounds,
                "status": ("RECOVERY_NOT_VERIFIED" if not audit_records[cell]["eligible"]
                           else "PASS" if result["passes"] else "CRITERIA_NOT_MET"),
            })

    fc.save_json(OUT / "SUMMARY.json", {
        "experiment": "wave2_fourroom_dim192",
        "family_size": family_size,
        "per_claim_alpha": alpha,
        "diagnostic_family_sizes": [9, 51],
        "rows": rows,
        "budget_rows": budget_rows,
        "projection_meta": projection_meta,
    })
    report = [
        "# Wave-2 S1: FourRoom Level-1 192-D readout projections", "",
        f"family_size {family_size}, alpha {alpha:.6g}.", "",
        "| Cell | Readout | Audit min | Coverage | Errors/Accepted | Upper | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        report.append(
            f"| {row['cell']} | {row['readout']} | {row['minimum_r2']:.3f} | "
            f"{row['coverage']:.3f} | {row['errors']}/{row['accepted']} | "
            f"{row['error_upper']:.4f} | {row['status']} |")
    report += ["", "grid4x4_pca192 is fit on TRAIN frames only; grid4x4_rand192 uses a "
               "fixed QR orthonormal projection.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(report) + "\n")

    by_budget = {}
    for row in budget_rows:
        key = f"{row['model']}|{row['readout']}|{row['n_train']}"
        by_budget.setdefault(key, []).append(row["audit_min_r2"])
    budget_summary = {key: {"mean_audit_min_r2": float(np.mean(values)),
                            "values": values}
                      for key, values in by_budget.items()}
    fc.save_json(OUT / "budget_curve_summary.json", budget_summary)
    budget_md = ["# S1-B sample-budget curve", "",
                 "| Model | Readout | n_train | Mean audit min R2 |", "|---|---|---:|---:|"]
    for key, value in budget_summary.items():
        model, readout, n_train = key.split("|")
        budget_md.append(f"| {model} | {readout} | {n_train} | "
                         f"{value['mean_audit_min_r2']:.4f} |")
    (OUT / "budget_curve_summary.md").write_text("\n".join(budget_md) + "\n")
    logging.info("S1 complete: %s cells, %s budget points -> %s", len(rows),
                 len(budget_rows), OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
