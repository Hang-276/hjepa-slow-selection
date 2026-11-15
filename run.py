"""Frozen visual H-JEPA: physical-factor recovery and selective slow-axis choice."""

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import hdf5plugin  # Registers the official collector's compression filter.
import numpy as np
import torch
from torch import nn

from metrics import risk, score_windows

ROOT = Path(__file__).resolve().parent
AXES = ["agent_x", "agent_y", "distractor_x", "distractor_y"]
ROLES = ("train", "val", "audit", "ref", "cal", "test")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(source, data, out, cfg):
    data.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(source / "scripts/data/collect_fourroom_distractors.py"),
               f"cache_dir={data}", "dataset_name=fourroom", f"seed={cfg['collection_seed']}",
               "num_traj=null", f"num_transitions={400 * sum(cfg['episodes'].values())}",
               f"ego_policy.mixture.0.params.action_repeat_prob={cfg['expert_action_repeat_prob']}",
               f"hydra.run.dir={out / 'collection'}", "hydra.job.chdir=false"]
    save_json(out / "collection_command.json", command)
    env = dict(os.environ, PYTHONPATH=str(source), SDL_VIDEODRIVER="dummy")
    logging.info("Generating %s fresh episodes; see collection.log", sum(cfg["episodes"].values()))
    with (out / "collection.log").open("w") as log:
        subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copy2(data / "fourroom.collection.yaml", out / "collection.yaml")
    return data / "fourroom.h5"


def split_frames(data, cfg, out):
    lengths, offsets = data["ep_len"][:], data["ep_offset"][:]
    if len(lengths) != sum(cfg["episodes"].values()) or np.any(lengths != 401):
        raise ValueError("Expected exactly the configured number of fresh, 401-frame episodes")
    if data["pixels"].shape[1:] != (224, 224, 3):
        raise ValueError("Expected official 224x224 RGB frames")
    rng = np.random.default_rng(cfg["split_seed"])
    order = rng.permutation(len(lengths))
    indices, manifest, cursor = {}, {}, 0
    for role in ROLES:
        count = cfg["episodes"][role]
        ids = np.sort(order[cursor:cursor + count])
        cursor += count
        if role in ("train", "val"):
            local = np.stack([np.sort(rng.choice(401, cfg["fit_frames_per_episode"], replace=False))
                              for _ in ids])
        else:
            starts = rng.integers(0, 402 - cfg["window_length"], size=count)
            local = starts[:, None] + np.arange(cfg["window_length"])
        indices[role] = offsets[ids, None] + local
        manifest[role] = {"episode_ids": ids.tolist(), "frame_indices": indices[role].tolist()}
    save_json(out / "splits.json", manifest)
    return indices


def states(data, indices):
    # Contiguous reads stay within one assigned episode and avoid huge HDF5 point lists.
    values = []
    for row in indices:
        start, stop = int(row[0]), int(row[-1]) + 1
        values.append(np.concatenate([data["xy"][start:stop], data["distractor0_xy"][start:stop]], axis=1)[row - start])
    return np.stack(values).astype(np.float64)


def load_backbone(path, device):
    # Only the pinned, public official checkpoint is accepted by the launcher.
    model = torch.load(path, map_location="cpu", weights_only=False).eval().to(device)
    model.requires_grad_(False)
    if model.num_levels != 2:
        raise ValueError("This protocol evaluates the official two-level model")
    for level in (1, 2):
        if int(getattr(model.get_level(level), "temporal_window_size", 1)) != 1:
            raise ValueError("Dense same-time probing requires a one-frame encoder at each level")
    return model


@torch.inference_mode()
def encode(model, data, indices, device, batch_size):
    # Matches upstream final_probing_decoding_eval._dense_encode_no_stride.
    frames = indices.ravel()
    result = [[], []]
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    for start in range(0, len(frames), batch_size):
        pixels = torch.from_numpy(data["pixels"][frames[start:start + batch_size]])
        pixels = pixels.to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255
        embedded = model.get_level(1).encode({"pixels": ((pixels - mean) / std)[:, None]}, key="pixels")["embed_0"]
        result[0].append(embedded[:, 0].cpu().numpy())
        embedded = model.get_level(2).encode({"embed_1": embedded}, key="embed_1")["embed_0"]
        result[1].append(embedded[:, 0].cpu().numpy())
        if start and start % (batch_size * 100) == 0:
            logging.info("Encoded %s / %s frames", start, len(frames))
    return [np.concatenate(part) for part in result]


def make_probe(dimension, hidden, device):
    return nn.Sequential(nn.Linear(dimension, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4)).to(device)


def fit_probe(train_x, train_y, val_x, val_y, seed, cfg, device, path):
    torch.manual_seed(seed)
    x_mean, x_std = train_x.mean(0), np.maximum(train_x.std(0), 1e-6)
    y_mean, y_std = train_y.mean(0), train_y.std(0)
    if np.any(y_std <= 0):
        raise ValueError("A training physical factor has zero variance")
    x = torch.as_tensor((train_x - x_mean) / x_std, device=device)
    y = torch.as_tensor((train_y - y_mean) / y_std, dtype=torch.float32, device=device)
    vx = torch.as_tensor((val_x - x_mean) / x_std, device=device)
    vy = torch.as_tensor((val_y - y_mean) / y_std, dtype=torch.float32, device=device)
    probe = make_probe(x.shape[1], cfg["probe_hidden"], device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=cfg["probe_learning_rate"],
                                 weight_decay=cfg["probe_weight_decay"])
    best_loss, best, trace = float("inf"), None, []
    for step in range(1, cfg["probe_steps"] + 1):
        ids = torch.randint(len(x), (cfg["probe_batch_size"],), device=device)
        loss = (probe(x[ids]) - y[ids]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite probe training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == cfg["probe_steps"]:
            with torch.no_grad():
                val_loss = (probe(vx) - vy).square().mean().item()
            trace.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss})
            if val_loss < best_loss:
                best_loss = val_loss
                best = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
    artifact = {"state_dict": best, "input_dim": x.shape[1], "hidden": cfg["probe_hidden"],
                "x_mean": torch.from_numpy(x_mean), "x_std": torch.from_numpy(x_std),
                "y_mean": torch.from_numpy(y_mean), "y_std": torch.from_numpy(y_std)}
    torch.save(artifact, path)
    save_json(path.with_suffix(".json"), {"seed": seed, "validation_loss": best_loss, "trace": trace})
    return artifact


@torch.inference_mode()
def predict(artifact, features, device):
    probe = make_probe(artifact["input_dim"], artifact["hidden"], device)
    probe.load_state_dict(artifact["state_dict"])
    probe.eval()
    mean, std = artifact["x_mean"].numpy(), artifact["x_std"].numpy()
    output = []
    for chunk in range(0, len(features), 4096):
        x = torch.as_tensor((features[chunk:chunk + 4096] - mean) / std, device=device)
        output.append(probe(x).cpu().numpy())
    return np.concatenate(output) * artifact["y_std"].numpy() + artifact["y_mean"].numpy()


def execute(args, cfg, out):
    source = ROOT / "assets/H-JEPA"
    actual_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual_commit != cfg["upstream_commit"]:
        raise ValueError("Upstream source revision does not match config.json; run setup.sh")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source):
        raise ValueError("Official tracked source has local changes")
    sys.path[:0] = [str(source / "h_jepa"), str(source)]
    weights = {seed: ROOT / f"assets/weights/fourroom/fourroom_hjepa_l2/seed{seed}/fourroom_hjepa_l2_object.ckpt"
               for seed in cfg["model_seeds"]}
    for path in weights.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run bash setup.sh")
    provenance = {"upstream_commit": actual_commit, "weights_sha256": {str(s): sha256(p) for s, p in weights.items()},
                  "python": sys.version, "torch": torch.__version__, "device": args.device,
                  "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None}
    setup_receipt = ROOT / "assets/setup.json"
    setup = json.loads(setup_receipt.read_text())
    if setup["upstream_commit"] != actual_commit or any(
            setup["weights_sha256"][seed] != digest for seed, digest in provenance["weights_sha256"].items()):
        raise ValueError("Official source or checkpoint differs from setup receipt; run setup.sh")
    provenance["setup"] = setup
    save_json(out / "provenance.json", provenance)
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=(out / "environment.txt").open("w"), check=True)
    data_path = collect(source, ROOT / "assets/data" / args.name, out, cfg)
    save_json(out / "data.json", {"path": str(data_path), "sha256": sha256(data_path)})
    family_size = len(weights) * 2 * 2
    alpha = cfg["family_alpha"] / family_size
    rows, calibration, probes, references, audits = [], {}, {}, {}, {}
    with h5py.File(data_path, "r") as data:
        indices = split_frames(data, cfg, out)
        physical_ref = states(data, indices["ref"])
        true_variance = physical_ref.reshape(-1, 4).var(axis=0)
        np.savez_compressed(out / "reference_truth.npz", states=physical_ref, variance=true_variance)
        for seed, path in weights.items():
            logging.info("Model seed %s: fitting readouts and freezing calibration", seed)
            model = load_backbone(path, args.device)
            features = {role: encode(model, data, indices[role], args.device, cfg["encode_batch_size"])
                        for role in ROLES[:-1]}
            for level in (1, 2):
                cell = f"seed{seed}_level{level}"
                folder = out / cell
                folder.mkdir()
                feature = {role: value[level - 1] for role, value in features.items()}
                artifact = fit_probe(feature["train"], states(data, indices["train"]).reshape(-1, 4),
                                     feature["val"], states(data, indices["val"]).reshape(-1, 4),
                                     seed * 10 + level, cfg, args.device, folder / "probe.pt")
                probes[cell] = artifact
                truth = states(data, indices["audit"]).reshape(-1, 4)
                audit_prediction = predict(artifact, feature["audit"], args.device)
                audit_variance = truth.var(0)
                if np.any(audit_variance <= 0):
                    raise ValueError("AUDIT factor has zero variance")
                r2 = 1 - np.mean((audit_prediction - truth)**2, axis=0) / audit_variance
                audits[cell] = {"r2": dict(zip(AXES, r2.tolist())), "eligible": bool(np.min(r2) >= cfg["audit_min_r2"])}
                save_json(folder / "audit.json", audits[cell])
                np.savez_compressed(folder / "audit.npz", prediction=audit_prediction, truth=truth)
                ref_prediction = predict(artifact, feature["ref"], args.device)
                ref_variance = ref_prediction.var(0)
                references[cell] = ref_variance
                np.savez_compressed(folder / "reference.npz", prediction=ref_prediction, variance=ref_variance)
                cal_prediction = predict(artifact, feature["cal"], args.device).reshape(*indices["cal"].shape, 4)
                # Threshold selection only uses predicted gaps; no CAL truth is loaded here.
                cal_scores = score_windows(cal_prediction, cal_prediction, ref_variance, ref_variance, cfg["lags"])
                calibration[cell] = float(np.quantile(cal_scores["gap"], cfg["cal_gap_quantile"]))
                np.savez_compressed(folder / "calibration.npz", prediction=cal_prediction,
                                    scores=cal_scores["predicted_scores"], gap=cal_scores["gap"])
                logging.info("%s AUDIT min R2=%.4f, CAL threshold=%.6f", cell, np.min(r2), calibration[cell])
            del features, model
        save_json(out / "thresholds.json", {"own_cal": calibration,
                  "level1_cal": {f"seed{s}": calibration[f"seed{s}_level1"] for s in weights},
                  "per_claim_alpha": alpha, "family_size": family_size})

        # All readouts, REF variances and thresholds are fixed before TEST encoding.
        for seed, path in weights.items():
            logging.info("Model seed %s: predicting sealed TEST windows", seed)
            model = load_backbone(path, args.device)
            features = encode(model, data, indices["test"], args.device, cfg["encode_batch_size"])
            for level in (1, 2):
                cell = f"seed{seed}_level{level}"
                prediction = predict(probes[cell], features[level - 1], args.device).reshape(*indices["test"].shape, 4)
                np.savez_compressed(out / cell / "test_prediction.npz", prediction=prediction)
            del features, model
        save_json(out / "forecast_receipt.json", {"thresholds_sha256": sha256(out / "thresholds.json"),
                  "prediction_sha256": {p.parent.name: sha256(p) for p in sorted(out.glob("*/test_prediction.npz"))}})
        test_truth = states(data, indices["test"])
    np.savez_compressed(out / "test_truth.npz", states=test_truth)
    for seed in weights:
        for level in (1, 2):
            cell = f"seed{seed}_level{level}"
            prediction = np.load(out / cell / "test_prediction.npz")["prediction"]
            scores = score_windows(prediction, test_truth, true_variance, references[cell], cfg["lags"])
            np.savez_compressed(out / cell / "test_scores.npz", **scores)
            for rule, threshold in (("own_cal", calibration[cell]), ("level1_cal", calibration[f"seed{seed}_level1"])):
                result = risk(scores["gap"], scores["wrong"], threshold, alpha,
                              cfg["minimum_coverage"], cfg["maximum_selective_error"])
                row = {"cell": cell, "rule": rule, "threshold": threshold,
                       "minimum_r2": min(audits[cell]["r2"].values()), "eligible": audits[cell]["eligible"], **result}
                row["status"] = ("SMOKE_ONLY" if args.smoke else "RECOVERY_NOT_VERIFIED" if not row["eligible"]
                                 else "PASS" if row["passes"] else "CRITERIA_NOT_MET")
                rows.append(row)
    save_json(out / "summary.json", {"engineering_smoke": args.smoke, "axes": AXES, "rows": rows})
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = ["# H-JEPA slow selection", "", "工程冒烟测试，不用于科学结论。" if args.smoke else
              "这是固定模型与本次数据分布上的独立测试结果；PASS 不表示一般性定理。", "",
              "| 模型/层 | 阈值来源 | 最低 R² | 接受比例 | 选错窗口/接受窗口 | 错误率上界 | 状态 |",
              "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        report.append(f"| {row['cell']} | {row['rule']} | {row['minimum_r2']:.3f} | {row['coverage']:.1%} | "
                      f"{row['errors']}/{row['accepted']} | {row['error_upper']:.2%} | {row['status']} |")
    report += ["", "RECOVERY_NOT_VERIFIED：本读出未通过四个因素的恢复检查，不能据此认定表征丢失了信息。",
               "CRITERIA_NOT_MET：恢复检查通过，但本次阈值未同时达到覆盖率与错误率要求。",
               "PASS：恢复检查通过，覆盖率至少 35%，且 Bonferroni 校正后的单侧错误率上界不超过 1%。",
               "覆盖率是测试集观测比例，不是总体覆盖率的置信下界。", "",
               "请将这个运行目录整体压缩发回；分析不需要 assets 中的图像或官方权重。"]
    (out / "RESULTS.md").write_text("\n".join(report) + "\n")
    save_json(out / "status.json", {"status": "SMOKE_COMPLETE" if args.smoke else "COMPLETE"})
    logging.info("Complete. Return the entire folder: %s", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Unique result folder name")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Engineering check only; never scientific evidence")
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        parser.error("Use only letters, digits, '_' and '-' in --name")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; check setup.sh and your NVIDIA driver")
    cfg = json.loads((ROOT / "config.json").read_text())
    if args.smoke:
        cfg.update(model_seeds=[42], collection_seed=2026111542, probe_steps=20, fit_frames_per_episode=8)
        cfg["episodes"] = dict.fromkeys(ROLES, 2)
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting a run; the shell may precreate launch.log.
    with (out / "config.json").open("x") as stream:
        json.dump({**cfg, "engineering_smoke": args.smoke, "axes": AXES}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["HF_HOME"] = str(ROOT / "assets/cache")
    snapshot = out / "scripts"
    snapshot.mkdir()
    for name in ("run.py", "metrics.py", "config.json", "README.md", "VALIDATION.md", "setup.sh"):
        shutil.copy2(ROOT / name, snapshot / name)
    save_json(out / "status.json", {"status": "RUNNING"})
    try:
        execute(args, cfg, out)
    except BaseException as exc:
        save_json(out / "status.json", {"status": "FAILED", "error": repr(exc)})
        logging.exception("Stopped; keep this result folder for diagnosis")
        raise


if __name__ == "__main__":
    main()
