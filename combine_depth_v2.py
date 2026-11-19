"""Combine the three Stage-3 runs into the pre-registered Family B depth table.

Family B (family_size = 51) at alpha = 0.05/51:
  LeWM   L1: 3 readouts x 3 seeds x own_cal                         =  9
  H-JEPA L2 L1: 3 readouts x 3 seeds x own_cal                     =  9
  H-JEPA L2 L2: cls_160k_official x 3 seeds x 2 rules              =  6
  H-JEPA L4 L1: 3 readouts x 3 seeds x own_cal                     =  9
  H-JEPA L4 L2-L4: cls_160k_official x 3 levels x 3 x 2 rules      = 18
                                                                  total 51

The script only reads sealed per-cell TEST scores (gap, wrong) and frozen
thresholds; it never reads physical TEST coordinates.
"""

import os

_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if _vis not in {"6", "7", "6,7", "7,6"}:
    raise RuntimeError(
        f"FATAL GPU ISOLATION GUARD: CUDA_VISIBLE_DEVICES={_vis!r}; only GPUs 6 and 7 are allowed!"
    )

import csv
import json
from pathlib import Path
import sys

import numpy as np

from metrics import risk

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "fourroom_depth_v2_combined"
FAMILY_SIZE = 51
ALPHA = 0.05 / FAMILY_SIZE
DIAGNOSTIC_SIZES = (9, 18, 27)
MIN_COVERAGE = 0.35
MAX_ERROR = 0.01
L1_READOUTS = ("cls_160k_official", "grid4x4_160k_official", "grid4x4_160k_ridge")
RUNS = {"lewm": "fourroom_lewm_v2", "l2": "fourroom_probe_v2", "l4": "fourroom_hjepa_l4_v2"}
SPEC = (
    ("lewm", 1, L1_READOUTS, ("own_cal",)),
    ("l2", 1, L1_READOUTS, ("own_cal",)),
    ("l2", 2, ("cls_160k_official",), ("own_cal", "level1_cal")),
    ("l4", 1, L1_READOUTS, ("own_cal",)),
    ("l4", 2, ("cls_160k_official",), ("own_cal", "level1_cal")),
    ("l4", 3, ("cls_160k_official",), ("own_cal", "level1_cal")),
    ("l4", 4, ("cls_160k_official",), ("own_cal", "level1_cal")),
)


def load_json(path):
    return json.loads(Path(path).read_text())


def main():
    run_state = {}
    for key, name in RUNS.items():
        run_dir = RESULTS / name
        status = load_json(run_dir / "status.json")
        if status.get("status") != "COMPLETE":
            raise ValueError(f"{name} is not COMPLETE: {status}")
        if not (run_dir / "forecast_receipt.json").is_file():
            raise FileNotFoundError(f"{name} has no sealed forecast receipt")
        run_state[key] = {
            "dir": run_dir,
            "thresholds": load_json(run_dir / "thresholds.json"),
            "config": load_json(run_dir / "config.json"),
        }

    rows, diagnostics = [], {}
    for model_key, level, readouts, rules in SPEC:
        state = run_state[model_key]
        for readout in readouts:
            for seed in state["config"]["model_seeds"]:
                cell = f"{readout}_seed{seed}_level{level}"
                cell_dir = state["dir"] / cell
                scores = np.load(cell_dir / "test_scores.npz")
                audit = load_json(cell_dir / "audit.json")
                threshold_cell = state["thresholds"]["cells"][cell]
                for rule in rules:
                    threshold = (threshold_cell if rule == "own_cal" else
                                 state["thresholds"]["level1_cal"][f"{readout}/seed{seed}"])
                    result = risk(scores["gap"], scores["wrong"], threshold, ALPHA,
                                  MIN_COVERAGE, MAX_ERROR)
                    bounds = {}
                    for size in DIAGNOSTIC_SIZES:
                        diagnostic = risk(scores["gap"], scores["wrong"], threshold,
                                          0.05 / size, MIN_COVERAGE, MAX_ERROR)
                        bounds[str(size)] = {
                            "per_claim_alpha": 0.05 / size,
                            "error_upper": diagnostic["error_upper"],
                            "passes": diagnostic["passes"],
                        }
                    row = {
                        "model": model_key,
                        "cell": cell,
                        "readout": readout,
                        "seed": seed,
                        "level": level,
                        "rule": rule,
                        "threshold": threshold,
                        "minimum_r2": float(min(audit["r2"].values())),
                        "eligible": bool(audit["eligible"]),
                        **result,
                        "family_size": FAMILY_SIZE,
                        "per_claim_alpha": ALPHA,
                        "diagnostic_error_upper": bounds,
                        "status": ("RECOVERY_NOT_VERIFIED" if not audit["eligible"]
                                   else "PASS" if result["passes"] else "CRITERIA_NOT_MET"),
                    }
                    rows.append(row)
                    per_cell = diagnostics.setdefault(cell, {
                        "cell": cell, "model": model_key, "level": level,
                        "readout": readout, "seed": seed,
                        "audit_r2": audit["r2"], "rules": {},
                    })
                    per_cell["rules"][rule] = {
                        "threshold": threshold,
                        "accepted": result["accepted"],
                        "errors": result["errors"],
                        "coverage": result["coverage"],
                        "selective_error": result["selective_error"],
                        "error_upper": result["error_upper"],
                        "status": row["status"],
                        "diagnostic_error_upper": bounds,
                    }
                cell_diagnostics = load_json(cell_dir / "diagnostics.json")
                diagnostics[cell]["window_groups"] = cell_diagnostics.get("window_groups", {})
                diagnostics[cell]["test_r2"] = cell_diagnostics.get("test_r2")
                diagnostics[cell]["predicted_over_physical_variance"] = \
                    cell_diagnostics.get("predicted_over_physical_variance")

    if len(rows) != FAMILY_SIZE:
        raise ValueError(f"Family B must have {FAMILY_SIZE} rows, got {len(rows)}")

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": "B_depth_lineage",
        "family_size": FAMILY_SIZE,
        "family_alpha": 0.05,
        "per_claim_alpha": ALPHA,
        "diagnostic_family_sizes": list(DIAGNOSTIC_SIZES),
        "row_count": len(rows),
        "sources": {key: str(state["dir"].relative_to(ROOT)) for key, state in run_state.items()},
        "rows": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# Family B: FourRoom depth lineage (L=1 / L=2 / L=4)", "",
        f"Family size {FAMILY_SIZE}, per-claim alpha {ALPHA:.6g}. "
        "Level-1 cells use own_cal only; Level >= 2 cells use own_cal and level1_cal.", "",
        "| Model | Level | Cell | Rule | Min R2 | Coverage | Errors/Accepted | Error upper | Status |",
        "|---|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        report.append(
            f"| {row['model']} | {row['level']} | {row['cell']} | {row['rule']} | "
            f"{row['minimum_r2']:.3f} | {row['coverage']:.1%} | "
            f"{row['errors']}/{row['accepted']} | {row['error_upper']:.2%} | {row['status']} |")
    report += [
        "",
        "RECOVERY_NOT_VERIFIED: AUDIT min R2 < 0.85; reported as a protocol-defined outcome.",
        "Level 2+ cells use the official CLS readout only; spatial-grid readouts exist at Level 1.",
        "Diagnostic bounds at M=9/18/27 are recorded per row in summary.json.",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {len(rows)} Family-B rows to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
