"""S2: 3x3 predictor factorial on the stationary two-sprite process.

data_seed = 2026111902, rho = (0.964, 0.961, 0.78, 0.76).
45 predictor models (lag {1,4,10} x w_sig {0.0005,0.004,0.02} x 5 seeds)
+ 5 symmetric_lejepa controls (lag=1, w_sig=0.05) = 50 models.
family_size = 50, alpha = 0.05/50; M=5 single-condition diagnostics.

    python3 wave2/s2_sprites_factorial.py --device cuda
    python3 wave2/s2_sprites_factorial.py --smoke --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sprites_core import (  # noqa: E402
    RHO_A, ROLE_SHAPES, SMOKE_SHAPES, generate_splits, pretrain_encoder,
    run_cell, save_json, set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_SEED = 2026111902
SEEDS = (201, 202, 203, 204, 205)
PREDICTOR_LAGS = (1, 4, 10)
PREDICTOR_W_SIG = (0.0005, 0.004, 0.02)
SYMMETRIC_W_SIG = 0.05


def conditions(smoke=False):
    if smoke:
        return [
            {"arm": "predictor", "lag": 1, "w_sig": 0.0005},
            {"arm": "symmetric", "lag": 1, "w_sig": SYMMETRIC_W_SIG},
        ]
    output = []
    for lag in PREDICTOR_LAGS:
        for w_sig in PREDICTOR_W_SIG:
            output.append({"arm": "predictor", "lag": lag, "w_sig": w_sig})
    output.append({"arm": "symmetric", "lag": 1, "w_sig": SYMMETRIC_W_SIG})
    return output


def summarize_condition(rows):
    audit = np.array([row["minimum_r2"] for row in rows], dtype=float)
    coverage = np.array([row["coverage"] for row in rows], dtype=float)
    errors = sum(row["errors"] for row in rows)
    accepted = sum(row["accepted"] for row in rows)
    regimes = {}
    for row in rows:
        regimes[row["regime"]] = regimes.get(row["regime"], 0) + 1
    return {
        "models": len(rows),
        "audit_min_r2_mean": float(audit.mean()),
        "audit_min_r2_min": float(audit.min()),
        "coverage_mean": float(coverage.mean()),
        "errors": int(errors),
        "accepted": int(accepted),
        "passes": int(sum(row["status"] == "PASS" for row in rows)),
        "recovery_not_verified": int(sum(row["status"] == "RECOVERY_NOT_VERIFIED" for row in rows)),
        "regimes": regimes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--pretrain-steps", type=int, default=None)
    parser.add_argument("--probe-steps", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    shapes = SMOKE_SHAPES if args.smoke else ROLE_SHAPES
    seeds = (SEEDS[0],) if args.smoke else SEEDS
    arms = conditions(args.smoke)
    pretrain_steps = args.pretrain_steps or (5 if args.smoke else 6000)
    probe_steps = args.probe_steps or (20 if args.smoke else 3000)
    out = Path(args.out or (ROOT / "results" /
                            ("wave2_sprites_factorial_smoke" if args.smoke
                             else "wave2_sprites_factorial")))
    out.mkdir(parents=True, exist_ok=True)
    logging.info("S2 data_seed=%s shapes=%s seeds=%s conditions=%s", DATA_SEED, shapes,
                 seeds, len(arms))
    data, _ = generate_splits(DATA_SEED, shapes=shapes)
    family_size = len(arms) * len(seeds)
    alpha = 0.05 / family_size
    rows = []
    for condition in arms:
        for seed in seeds:
            cell = (f"{condition['arm']}_lag{condition['lag']}_wsig{condition['w_sig']}"
                    f"_seed{seed}")
            logging.info("S2 cell %s", cell)
            set_seed(seed)
            encoder, _, trace = pretrain_encoder(
                data["pretrain"], mode=("symmetric_lejepa" if condition["arm"] == "symmetric"
                                        else "predictor"),
                lag=condition["lag"], w_sig=condition["w_sig"], seed=seed,
                device=args.device, steps=pretrain_steps, batch=(64 if args.smoke else 256),
                log_every=max(1, pretrain_steps // 4))
            (out / cell).mkdir(parents=True, exist_ok=True)
            save_json(out / cell / "pretrain.json",
                      {"mode": condition["arm"], "lag": condition["lag"],
                       "w_sig": condition["w_sig"], "seed": seed, "trace": trace})
            row, diagnostics, _ = run_cell(
                encoder, data, out_dir=out / cell, cell=cell, seed=seed, alpha=alpha,
                device=args.device, probe_steps=probe_steps,
                probe_batch=(64 if args.smoke else 256))
            row.update({k: condition[k] for k in ("arm", "lag", "w_sig")})
            row["regime"] = diagnostics["regime"]
            row["family_size"] = family_size
            row["diagnostic_M5_upper"] = None
            from metrics import risk  # local import keeps module import order simple
            row["diagnostic_M5_upper"] = risk(
                np.load(out / cell / "test_scores.npz")["gap"],
                np.load(out / cell / "test_scores.npz")["wrong"],
                row["threshold"], 0.05 / 5, 0.35, 0.01)["error_upper"]
            rows.append(row)
            logging.info("S2 %s audit=%.4f cov=%.3f err=%s/%s status=%s regime=%s",
                         cell, row["minimum_r2"], row["coverage"], row["errors"],
                         row["accepted"], row["status"], row["regime"])

    per_condition = {}
    for condition in arms:
        key = f"{condition['arm']}_lag{condition['lag']}_wsig{condition['w_sig']}"
        selected = [row for row in rows if row["arm"] == condition["arm"]
                    and row["lag"] == condition["lag"] and row["w_sig"] == condition["w_sig"]]
        summary = summarize_condition(selected)
        summary.update({"arm": condition["arm"], "lag": condition["lag"],
                        "w_sig": condition["w_sig"]})
        per_condition[key] = summary
    save_json(out / "SUMMARY.json", {
        "experiment": "wave2_sprites_factorial",
        "data_seed": DATA_SEED,
        "rho": list(RHO_A),
        "family_size": family_size,
        "family_alpha": 0.05,
        "per_claim_alpha": alpha,
        "diagnostic_family_size": 5,
        "rows": rows,
        "per_condition": per_condition,
    })
    report = [
        "# Wave-2 S2: two-sprite predictor factorial", "",
        f"data_seed {DATA_SEED}, rho {list(RHO_A)}. family_size {family_size}, "
        f"alpha {alpha:.6g}.", "",
        "| Arm | Lag | w_sig | Models | Audit min (mean/min) | Coverage | Errors/Accepted | PASS | Regimes |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for key, value in per_condition.items():
        report.append(
            f"| {value['arm']} | {value['lag']} | {value['w_sig']} | {value['models']} | "
            f"{value['audit_min_r2_mean']:.3f}/{value['audit_min_r2_min']:.3f} | "
            f"{value['coverage_mean']:.3f} | {value['errors']}/{value['accepted']} | "
            f"{value['passes']} | {value['regimes']} |")
    report += ["", "PASS requires AUDIT min R2 >= 0.85, coverage >= 0.35, and the "
               "Bonferroni-corrected 1% error bound.", ""]
    (out / "RESULTS.md").write_text("\n".join(report) + "\n")
    logging.info("S2 complete: %s rows -> %s", len(rows), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
