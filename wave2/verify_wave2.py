"""Wave-2 integrity check, unified summary, and tarball builder.

Verifies every per-cell forecast_receipt.json (and S1's global receipt), checks
all .npz payloads for NaN/Inf, writes
results/wave2_combined_summary/WAVE2_FINAL_SUMMARY.md and
wave2_integrity_check.json, and creates:

  wave2_summary_light.tar.gz  (< 5 MB: code + .md/.json/.csv summaries)
  wave2_full_bundle.tar.gz    (adds test/audit/calibration npz + probe*.pt)

Binaries are never committed to git.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUNS = ("wave2_sprites_factorial", "wave2_sprites_regime_switch",
        "wave2_fourroom_dim192")
LIGHT = ROOT / "wave2_summary_light.tar.gz"
FULL = ROOT / "wave2_full_bundle.tar.gz"
SELECTED_BINARIES = ("test_prediction.npz", "test_scores.npz", "audit.npz",
                     "reference.npz", "calibration.npz")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def find_probe(folder):
    for name in ("probe.pt", "probe_official.pt", "probe_ridge.pt"):
        if (folder / name).is_file():
            return folder / name
    return None


def check_npz(path):
    problems = []
    try:
        with np.load(path) as payload:
            for key in payload.files:
                array = payload[key]
                if array.dtype.kind in "fc" and not np.isfinite(array).all():
                    problems.append(f"{path.relative_to(ROOT)}:{key}")
    except Exception as error:  # pragma: no cover - diagnostic
        problems.append(f"{path.relative_to(ROOT)}: unreadable ({error})")
    return problems


def verify_run(run):
    directory = RESULTS / run
    report = {"receipts": 0, "npz_files": 0, "problems": [], "missing": False}
    if not directory.is_dir():
        report["missing"] = True
        report["problems"].append("run directory missing")
        return report
    receipts = sorted(directory.glob("*/forecast_receipt.json"))
    global_receipt = directory / "forecast_receipt.json"
    if global_receipt.is_file():
        receipts.append(global_receipt)
    for receipt_path in receipts:
        folder = receipt_path.parent
        receipt = load_json(receipt_path)
        report["receipts"] += 1
        threshold = receipt.get("cell_thresholds_sha256") or receipt.get("thresholds_sha256")
        if threshold is not None:
            thresholds_path = folder / "thresholds.json"
            if not thresholds_path.is_file() or sha256(thresholds_path) != threshold:
                report["problems"].append(
                    f"{receipt_path.relative_to(ROOT)}: thresholds hash mismatch")
        if "probe_sha256" in receipt:
            probe = find_probe(folder)
            if probe is None or sha256(probe) != receipt["probe_sha256"]:
                report["problems"].append(
                    f"{receipt_path.relative_to(ROOT)}: probe hash mismatch")
        if "test_prediction_sha256" in receipt and isinstance(
                receipt["test_prediction_sha256"], str):
            prediction = folder / "test_prediction.npz"
            if not prediction.is_file() or sha256(prediction) != receipt["test_prediction_sha256"]:
                report["problems"].append(
                    f"{receipt_path.relative_to(ROOT)}: prediction hash mismatch")
        if isinstance(receipt.get("test_prediction_sha256"), dict):
            for cell, digest in receipt["test_prediction_sha256"].items():
                prediction = directory / cell / "test_prediction.npz"
                if not prediction.is_file() or sha256(prediction) != digest:
                    report["problems"].append(
                        f"{receipt_path.relative_to(ROOT)}: prediction hash mismatch for {cell}")
    for path in directory.rglob("*.npz"):
        report["npz_files"] += 1
        report["problems"].extend(check_npz(path))
    summary = directory / "SUMMARY.json"
    if summary.is_file():
        report["summary_rows"] = len(load_json(summary).get("rows", []))
    else:
        report["problems"].append("SUMMARY.json missing")
    return report


def markdown_tables():
    lines = ["# Wave-2 combined summary", ""]
    s1 = RESULTS / "wave2_fourroom_dim192/SUMMARY.json"
    if s1.is_file():
        payload = load_json(s1)
        lines += ["## S1: FourRoom Level-1 192-D projections", "",
                  f"family_size {payload['family_size']}, alpha "
                  f"{payload['per_claim_alpha']:.6g}.", "",
                  "| Cell | Readout | Audit min | Coverage | Errors/Accepted | Upper | Status |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for row in payload["rows"]:
            lines.append(
                f"| {row['cell']} | {row['readout']} | {row['minimum_r2']:.3f} | "
                f"{row['coverage']:.3f} | {row['errors']}/{row['accepted']} | "
                f"{row['error_upper']:.4f} | {row['status']} |")
        lines.append("")
    s2 = RESULTS / "wave2_sprites_factorial/SUMMARY.json"
    if s2.is_file():
        payload = load_json(s2)
        lines += ["## S2: two-sprite predictor factorial", "",
                  f"family_size {payload['family_size']}, alpha "
                  f"{payload['per_claim_alpha']:.6g}.", "",
                  "| Arm | Lag | w_sig | Models | Audit mean/min | Coverage | Errors/Accepted | PASS | Regimes |",
                  "|---|---:|---:|---:|---|---:|---:|---:|---|"]
        for value in payload["per_condition"].values():
            lines.append(
                f"| {value['arm']} | {value['lag']} | {value['w_sig']} | "
                f"{value['models']} | {value['audit_min_r2_mean']:.3f}/"
                f"{value['audit_min_r2_min']:.3f} | {value['coverage_mean']:.3f} | "
                f"{value['errors']}/{value['accepted']} | {value['passes']} | "
                f"{value['regimes']} |")
        lines.append("")
    s3 = RESULTS / "wave2_sprites_regime_switch/SUMMARY.json"
    if s3.is_file():
        payload = load_json(s3)
        lines += ["## S3: episode-level regime switch", "",
                  f"family_size {payload['family_size']}, alpha "
                  f"{payload['per_claim_alpha']:.6g}.", "",
                  "| Arm | Models | Audit mean | PASS | Mode-B coverage | Regimes |",
                  "|---|---:|---:|---:|---:|---|"]
        for name, value in payload["per_arm"].items():
            lines.append(
                f"| {name} | {value['models']} | {value['audit_min_mean']:.3f} | "
                f"{value['passes']} | {value['mode_b_coverage_mean']:.3f} | "
                f"{value['regimes']} |")
        lines.append("")
    lines += ["## Integrity", "",
              "Per-cell forecast receipts and every .npz payload are checked by "
              "verify_wave2.py. Binaries are excluded from git.", ""]
    return "\n".join(lines)


def add_tree(tar, root, predicate):
    for path in sorted(root.rglob("*")):
        if path.is_file() and predicate(path):
            tar.add(path, arcname=str(path.relative_to(ROOT)))


def main():
    combined = RESULTS / "wave2_combined_summary"
    combined.mkdir(parents=True, exist_ok=True)
    reports = {run: verify_run(run) for run in RUNS}
    all_ok = all(not report["problems"] for report in reports.values())
    payload = {"runs": reports, "all_ok": all_ok}
    (combined / "wave2_integrity_check.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    (combined / "WAVE2_FINAL_SUMMARY.md").write_text(markdown_tables())

    code_predicate = lambda path: ("__pycache__" not in path.parts
                                   and path.suffix != ".pyc")
    with tarfile.open(LIGHT, "w:gz") as tar:
        add_tree(tar, ROOT / "wave2", code_predicate)
        add_tree(tar, combined, lambda path: True)
        for run in RUNS:
            directory = RESULTS / run
            if directory.is_dir():
                add_tree(tar, directory,
                         lambda path: path.suffix in (".md", ".json", ".csv"))
    with tarfile.open(FULL, "w:gz") as tar:
        add_tree(tar, ROOT / "wave2", code_predicate)
        add_tree(tar, combined, lambda path: True)
        for run in RUNS:
            directory = RESULTS / run
            if directory.is_dir():
                add_tree(tar, directory,
                         lambda path: (path.suffix in (".md", ".json", ".csv")
                                       or path.name in SELECTED_BINARIES
                                       or (path.name.startswith("probe")
                                           and path.suffix == ".pt")))
    sizes = {}
    for path in (LIGHT, FULL):
        sizes[path.name] = path.stat().st_size
        print(f"{path.name} bytes={sizes[path.name]} sha256={sha256(path)}")
    light_bytes = sizes[LIGHT.name]
    if light_bytes >= 5 * 1024 * 1024:
        print(f"ERROR: {LIGHT.name} is {light_bytes} bytes, over the 5 MiB light limit",
              file=sys.stderr)
        all_ok = False
    print(f"all_ok={all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
