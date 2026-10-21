import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ai.model import DriftRegressor
from edge_ai.experiment import (
    environment, load_dataset, metrics, scale_inputs, sha256,
    validate_checkpoint, write_json,
)


def fit_linear(X, y, alpha=0.0):
    """Intercept + full-window features; ridge does not penalize intercept."""
    design = np.column_stack((np.ones(len(X)), X.reshape(len(X), -1))).astype(np.float64)
    if alpha == 0:
        return np.linalg.lstsq(design, y, rcond=None)[0]
    penalty = np.eye(design.shape[1]) * np.sqrt(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.lstsq(
        np.vstack((design, penalty)), np.concatenate((y, np.zeros(design.shape[1]))),
        rcond=None,
    )[0]


def linear_predict(X, weights):
    design = np.column_stack((np.ones(len(X)), X.reshape(len(X), -1)))
    return design @ weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_dataset.npz")
    parser.add_argument("--model", default="artifacts/model.pt")
    parser.add_argument("--output", default="results/evaluation.json")
    args = parser.parse_args()
    data_path, model_path = ROOT / args.data, ROOT / args.model
    X, y, ids, metadata = load_dataset(data_path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint, data_path, ids)
    masks = {name: np.isin(ids, runs) for name, runs in checkpoint["split"].items()}
    scaled = scale_inputs(X)
    model = DriftRegressor(checkpoint["window_size"], checkpoint["channels"], checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        neural = model(torch.from_numpy(scaled[masks["test"]])).numpy()
    train, val, test = (masks[name] for name in ("train", "validation", "test"))
    ols = fit_linear(scaled[train], y[train])
    candidates = []
    for alpha in (1e-6, 1e-4, 1e-2, 1.0):
        weights = fit_linear(scaled[train], y[train], alpha)
        score = metrics(y[val], linear_predict(scaled[val], weights))["rmse_mm"]
        candidates.append((score, alpha, weights))
    _, best_alpha, ridge = min(candidates, key=lambda item: item[0])
    predictions = {
        "neural": neural,
        "last_value_proxy": 0.010 * X[test, -1, 0] + 0.004 * X[test, -1, 1],
        "linear_full_window": linear_predict(scaled[test], ols),
        "ridge_full_window": linear_predict(scaled[test], ridge),
    }
    scores = {name: metrics(y[test], prediction) for name, prediction in predictions.items()}
    report = {
        "environment": environment(), "dataset_sha256": sha256(data_path),
        "model_sha256": sha256(model_path), "dataset_metadata": metadata,
        "split": checkpoint["split"],
        "sample_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "ridge": {"selected_alpha": best_alpha,
                  "selection": "minimum validation RMSE; fit on training runs only",
                  "candidates": [{"alpha": a, "validation_rmse_mm": s} for s, a, _ in candidates]},
        "metrics": scores,
    }
    output = ROOT / args.output
    write_json(output, report)
    np.savez_compressed(output.with_suffix(".predictions.npz"),
                        target_mm=y[test], run_ids=ids[test], **predictions)
    for name, score in scores.items():
        print(f"{name}: MAE={score['mae_mm']:.9f} mm; RMSE={score['rmse_mm']:.9f} mm")
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()
