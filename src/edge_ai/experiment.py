"""Shared experiment contracts and provenance for synthetic drift forecasting."""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    root = Path(__file__).resolve().parents[2]
    source_paths = [root / "CMakeLists.txt", root / "requirements.txt"]
    for folder, pattern in (("scripts", "*.py"), ("src", "*.py"), ("tests", "*.py"), ("cpp", "*.cpp")):
        source_paths.extend((root / folder).rglob(pattern))
    packages = {}
    for name in ("numpy", "torch", "onnx", "onnxruntime", "onnxscript", "pytest", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2],
            text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "source_commit": commit, "source_dirty": dirty,
        "source_files_sha256": {
            path.relative_to(root).as_posix(): sha256(path)
            for path in sorted(source_paths) if path.is_file()
        },
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def split_runs(run_ids, seed):
    runs = np.unique(run_ids)
    np.random.default_rng(seed).shuffle(runs)
    n_train, n_val = int(0.70 * len(runs)), int(0.15 * len(runs))
    if min(n_train, n_val, len(runs) - n_train - n_val) < 1:
        raise ValueError("Need enough runs for nonempty train/validation/test splits")
    return {
        "train": runs[:n_train].tolist(),
        "validation": runs[n_train:n_train + n_val].tolist(),
        "test": runs[n_train + n_val:].tolist(),
    }


def validate_split(split, run_ids):
    groups = [set(split[name]) for name in ("train", "validation", "test")]
    if any(not group for group in groups):
        raise ValueError("Every split must contain runs")
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Run overlap between splits")
    if set.union(*groups) != set(np.unique(run_ids).tolist()):
        raise ValueError("Split does not match dataset runs")


def load_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        X, y, ids = data["X"].copy(), data["y"].copy(), data["run_ids"].copy()
        metadata = json.loads(str(data["metadata"])) if "metadata" in data else None
    if X.ndim != 3 or X.shape[1:] != (32, 2):
        raise ValueError("Expected input shape [samples, 32, 2]")
    if len(X) == 0 or y.shape != (len(X),) or ids.shape != (len(X),):
        raise ValueError("Invalid target/run ID shape or empty dataset")
    if not all(np.isfinite(a).all() for a in (X, y, ids)):
        raise ValueError("Dataset contains non-finite values")
    return X.astype(np.float32), y.astype(np.float32), ids, metadata


def validate_checkpoint(checkpoint, data_path, ids):
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Legacy checkpoint has no verified split metadata. Retrain with scripts/train.py.")
    if checkpoint["dataset_sha256"] != sha256(data_path):
        raise ValueError("Dataset hash differs from training data. Regenerate the original data or retrain.")
    validate_split(checkpoint["split"], ids)
    if checkpoint["preprocessing"] != {"temperature_divisor": 10.0, "vibration_divisor": 1.0}:
        raise ValueError("Unsupported preprocessing contract")


def scale_inputs(X):
    scaled = np.array(X, dtype=np.float32, copy=True)
    scaled[:, :, 0] /= 10.0
    return scaled


def metrics(target, prediction):
    error = np.asarray(prediction, dtype=np.float64) - target
    return {
        "mae_mm": float(np.mean(np.abs(error))),
        "rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_error_mm": float(np.max(np.abs(error))),
    }
