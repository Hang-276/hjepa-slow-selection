"""S3: episode-level regime-switch mixture (80% Mode A / 20% Mode B).

data_seed = 2026111903. Every split independently draws 80% Mode A
(rho = 0.964,0.961,0.78,0.76; slow {0,1}) and 20% Mode B
(rho = 0.78,0.76,0.964,0.961; slow {2,3}). 15 models = 5 seeds x 3 arms.
family_size = 15, alpha = 0.05/15.

    python3 wave2/s3_sprites_regime_switch.py --device cuda
    python3 wave2/s3_sprites_regime_switch.py --smoke --device cpu
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
    RHO_A, RHO_B, ROLE_SHAPES, SMOKE_SHAPES, generate_splits, pretrain_encoder,
    run_cell, save_json, set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_SEED = 2026111903
SEEDS = (301, 302, 303, 304, 305)
ARMS = (
    {"arm": "symmetric_lejepa_lag1", "mode": "symmetric_lejepa", "lag": 1, "w_sig": 0.05},
    {"arm": "predictor_lag1", "mode": "predictor", "lag": 1, "w_sig": 0.02},
    {"arm": "predictor_lag10", "mode": "predictor", "lag": 10, "w_sig": 0.004},
)


def _stratum(gap, wrong, true_top2, true_gap, predicted_top2, threshold, mask):
    count = int(mask.sum())
    if count == 0:
        return {"windows": 0, "all_answer_error": None, "coverage": 0.0,
                "accepted": 0, "errors": 0, "selective_error": None,
                "constant_01_baseline_error": None}
    accept = (gap >= threshold) & mask
    accepted = int(accept.sum())
    errors = int((wrong & accept).sum())
    baseline = np.asarray([0, 1])
    baseline_wrong = (true_top2 != baseline).any(axis=1)
    return {
        "windows": count,
        "all_answer_error": float(wrong[mask].mean()),
        "coverage": accepted / count,
        "accepted": accepted,
        "errors": errors,
        "selective_error": errors / accepted if accepted else None,
        "constant_01_baseline_error": float(baseline_wrong[mask].mean()),
    }


def stratify(extras, threshold):
    gap = extras["gap"]
    wrong = extras["wrong"]
    true_top2 = extras["true_top2"]
    true_gap = extras["true_gap"]
    predicted_top2 = extras["predicted_top2"]
    modes = extras["modes"]
    if modes is None:
        raise ValueError("S3 requires per-test-episode modes")
    mode_a = modes == 0
    mode_b = modes == 1
    output = {
        "overall": _stratum(gap, wrong, true_top2, true_gap, predicted_top2,
                            threshold, np.ones_like(mode_a, dtype=bool)),
        "mode_a": _stratum(gap, wrong, true_top2, true_gap, predicted_top2,
                           threshold, mode_a),
        "mode_b": _stratum(gap, wrong, true_top2, true_gap, predicted_top2,
                           threshold, mode_b),
        "mode_b_true_gap": {
            "mean": float(true_gap[mode_b].mean()) if mode_b.any() else None,
            "min": float(true_gap[mode_b].min()) if mode_b.any() else None,
            "fraction_ge_0.25": float((true_gap[mode_b] >= 0.25).mean()) if mode_b.any() else None,
            "fraction_ge_0.50": float((true_gap[mode_b] >= 0.50).mean()) if mode_b.any() else None,
        },
    }
    for zone in (0.25, 0.50):
        mask = true_gap >= zone
        output[f"indifference_ge_{zone:.2f}"] = _stratum(
            gap, wrong, true_top2, true_gap, predicted_top2, threshold, mask)
        output[f"indifference_mode_b_ge_{zone:.2f}"] = _stratum(
            gap, wrong, true_top2, true_gap, predicted_top2, threshold, mask & mode_b)
    return output


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
    arms = ARMS
    pretrain_steps = args.pretrain_steps or (5 if args.smoke else 6000)
    probe_steps = args.probe_steps or (20 if args.smoke else 3000)
    out = Path(args.out or (ROOT / "results" /
                            ("wave2_sprites_regime_switch_smoke" if args.smoke
                             else "wave2_sprites_regime_switch")))
    out.mkdir(parents=True, exist_ok=True)
    logging.info("S3 data_seed=%s shapes=%s seeds=%s arms=%s", DATA_SEED, shapes,
                 seeds, len(arms))
    data, modes = generate_splits(
        DATA_SEED, shapes=shapes,
        mixture={"p_b": 0.2, "rho_a": RHO_A, "rho_b": RHO_B})
    data["_test_modes"] = modes["test"]
    family_size = len(arms) * len(seeds)
    alpha = 0.05 / family_size
    rows, stratifications = [], {}
    for arm in arms:
        for seed in seeds:
            cell = f"{arm['arm']}_seed{seed}"
            logging.info("S3 cell %s", cell)
            set_seed(seed)
            encoder, _, trace = pretrain_encoder(
                data["pretrain"], mode=arm["mode"], lag=arm["lag"], w_sig=arm["w_sig"],
                seed=seed, device=args.device, steps=pretrain_steps,
                batch=(64 if args.smoke else 256),
                log_every=max(1, pretrain_steps // 4))
            (out / cell).mkdir(parents=True, exist_ok=True)
            save_json(out / cell / "pretrain.json", {
                "mode": arm["mode"], "lag": arm["lag"], "w_sig": arm["w_sig"],
                "seed": seed, "trace": trace, "mixture": {"p_b": 0.2,
                "rho_a": list(RHO_A), "rho_b": list(RHO_B)}})
            row, diagnostics, extras = run_cell(
                encoder, data, out_dir=out / cell, cell=cell, seed=seed, alpha=alpha,
                device=args.device, probe_steps=probe_steps,
                probe_batch=(64 if args.smoke else 256))
            stratified = stratify(extras, row["threshold"])
            save_json(out / cell / "stratified.json", stratified)
            diagnostics["stratified"] = stratified
            save_json(out / cell / "diagnostics.json", diagnostics)
            row.update({"arm": arm["arm"], "lag": arm["lag"], "w_sig": arm["w_sig"],
                        "regime": diagnostics["regime"], "family_size": family_size,
                        "mode_b_coverage": stratified["mode_b"]["coverage"],
                        "mode_b_selective_error": stratified["mode_b"]["selective_error"]})
            rows.append(row)
            stratifications[cell] = stratified
            logging.info("S3 %s audit=%.4f status=%s modeB_cov=%.3f modeB_err=%s",
                         cell, row["minimum_r2"], row["status"],
                         row["mode_b_coverage"], row["mode_b_selective_error"])

    per_arm = {}
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm["arm"]]
        per_arm[arm["arm"]] = {
            "models": len(selected),
            "audit_min_mean": float(np.mean([row["minimum_r2"] for row in selected])),
            "passes": int(sum(row["status"] == "PASS" for row in selected)),
            "mode_b_coverage_mean": float(np.mean([row["mode_b_coverage"] for row in selected])),
            "mode_b_selective_errors": [
                row["mode_b_selective_error"] for row in selected],
            "overall_strata": [stratifications[row["cell"]]["overall"] for row in selected],
            "mode_a_strata": [stratifications[row["cell"]]["mode_a"] for row in selected],
            "mode_b_strata": [stratifications[row["cell"]]["mode_b"] for row in selected],
            "regimes": {name: sum(row["regime"] == name for row in selected)
                        for name in ("Normal Recovery", "Regime II-A", "Regime II-B", "Mixed")},
        }
    save_json(out / "SUMMARY.json", {
        "experiment": "wave2_sprites_regime_switch",
        "data_seed": DATA_SEED,
        "mixture": {"p_b": 0.2, "rho_a": list(RHO_A), "rho_b": list(RHO_B)},
        "family_size": family_size,
        "family_alpha": 0.05,
        "per_claim_alpha": alpha,
        "rows": rows,
        "per_arm": per_arm,
        "stratifications": stratifications,
    })
    report = [
        "# Wave-2 S3: episode-level regime switch", "",
        f"data_seed {DATA_SEED}; 80% Mode A / 20% Mode B. family_size {family_size}, "
        f"alpha {alpha:.6g}.", "",
        "| Arm | Models | Audit min mean | PASS | Mode-B coverage | Mode-B selective error | Regimes |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for name, value in per_arm.items():
        errors = [e for e in value["mode_b_selective_errors"] if e is not None]
        report.append(
            f"| {name} | {value['models']} | {value['audit_min_mean']:.3f} | "
            f"{value['passes']} | {value['mode_b_coverage_mean']:.3f} | "
            f"{np.mean(errors):.3f} | {value['regimes']} |" if errors else
            f"| {name} | {value['models']} | {value['audit_min_mean']:.3f} | "
            f"{value['passes']} | {value['mode_b_coverage_mean']:.3f} | n/a | "
            f"{value['regimes']} |")
    report += ["", "Mode-B coverage > 0 confirms active switching (not the 0/0 abstention "
               "case). Constant-{0,1} baseline is in each cell's stratified.json.", ""]
    (out / "RESULTS.md").write_text("\n".join(report) + "\n")
    logging.info("S3 complete: %s rows -> %s", len(rows), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
