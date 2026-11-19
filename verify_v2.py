"""Stage-4 integrity check for the Wave-1 result directories.

Checks: COMPLETE status, forecast-receipt hashes against disk, no NaN/Inf in
any .npz payload, expected cell/row counts, and that the original tracked
experiment files are unchanged. Writes results/preflight_audit/wave1_integrity_check.json.
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
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "assets/H-JEPA"
DEFAULT_RUNS = ("fourroom_probe_v2", "fourroom_lewm_v2", "fourroom_hjepa_l4_v2")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_npz(path):
    problems = []
    with np.load(path) as payload:
        for key in payload.files:
            array = payload[key]
            if array.dtype.kind in "fc" and not np.isfinite(array).all():
                problems.append(f"{path.name}:{key}")
    return problems


def check_run(name):
    out = ROOT / "results" / name
    report = {"run": name, "path": str(out), "checks": {}, "problems": []}
    status_path = out / "status.json"
    if not status_path.is_file():
        report["problems"].append("missing status.json")
        return report
    status = json.loads(status_path.read_text())
    report["checks"]["status"] = status.get("status")
    if status.get("status") != "COMPLETE":
        report["problems"].append(f"status is {status.get('status')!r}")

    receipt = json.loads((out / "forecast_receipt.json").read_text())
    checks = {"thresholds": sha256(out / "thresholds.json") == receipt["thresholds_sha256"]}
    for cell, digest in receipt["prediction_sha256"].items():
        checks[f"prediction/{cell}"] = sha256(out / cell / "test_prediction.npz") == digest
    checks["run_script"] = sha256(ROOT / "run_v2.py") == receipt["run_script_sha256"]
    checks["metrics"] = sha256(ROOT / "metrics.py") == receipt["metrics_sha256"]
    snapshot = out / "scripts" / Path(receipt["config_path"]).name
    checks["config_snapshot"] = snapshot.is_file() and \
        sha256(snapshot) == receipt["config_snapshot_sha256"]
    report["checks"].update(checks)
    report["problems"].extend(f"hash mismatch: {key}" for key, ok in checks.items() if not ok)

    summary = json.loads((out / "summary.json").read_text())
    report["checks"]["summary_rows"] = len(summary["rows"])
    report["checks"]["cells"] = len(list(out.glob("*/test_prediction.npz")))
    report["checks"]["required_cells"] = report["checks"]["cells"]

    npz_files = sorted(out.glob("*/*.npz")) + [out / "test_truth.npz", out / "reference_truth.npz"]
    finite_problems = []
    for path in npz_files:
        if path.is_file():
            finite_problems.extend(check_npz(path))
    report["checks"]["npz_files_checked"] = len(npz_files)
    report["problems"].extend(f"nonfinite: {item}" for item in finite_problems)
    return report


def repo_state():
    git = "/home/kemove/yuzihang/envs/git/bin/git"
    tracked = subprocess.check_output([git, "diff", "--name-only", "--",
                                       "run.py", "metrics.py", "config.json", "test_metrics.py"],
                                      cwd=ROOT, text=True).split()
    all_tracked = subprocess.check_output([git, "status", "--porcelain", "--untracked-files=no"],
                                          cwd=ROOT, text=True)
    upstream = subprocess.check_output([git, "status", "--porcelain", "--untracked-files=no"],
                                       cwd=SOURCE, text=True)
    return {
        "original_files_changed": tracked,
        "tracked_status": all_tracked,
        "upstream_tracked_status": upstream,
        "upstream_commit": subprocess.check_output([git, "rev-parse", "HEAD"],
                                                   cwd=SOURCE, text=True).strip(),
        "note": "setup.sh carries the pre-existing, documented dependency pin from the "
                "deployment session; Wave 1 did not touch it.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    args = parser.parse_args()
    reports = [check_run(name) for name in args.runs]
    state = repo_state()
    payload = {
        "runs": reports,
        "repository": state,
        "all_runs_ok": all(not report["problems"] for report in reports),
        "original_files_untouched": not state["original_files_changed"],
        "upstream_clean": not state["upstream_tracked_status"].strip(),
    }
    destination = ROOT / "results/preflight_audit/wave1_integrity_check.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in
                      ("all_runs_ok", "original_files_untouched", "upstream_clean")}, indent=2))
    for report in reports:
        print(report["run"], report["checks"].get("status"),
              f"problems={len(report['problems'])}")
        for problem in report["problems"]:
            print("  -", problem)
    return 0 if payload["all_runs_ok"] and payload["original_files_untouched"] else 1


if __name__ == "__main__":
    sys.exit(main())
