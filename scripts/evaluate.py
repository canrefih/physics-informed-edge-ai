import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ai.model import DriftRegressor


def report_metrics(name, target, prediction):
    error = prediction - target
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))

    print(f"\n{name}")
    print(f"  MAE:           {mae:.6f} mm")
    print(f"  RMSE:          {rmse:.6f} mm")
    print(f"  Max abs error: {max_error:.6f} mm")


def main():
    data = np.load(ROOT / "data/synthetic_dataset.npz")
    X, y, run_ids = data["X"], data["y"], data["run_ids"]

    # Must match the split used in train.py.
    unique_runs = np.unique(run_ids)
    rng = np.random.default_rng(42)
    rng.shuffle(unique_runs)

    n_train = int(0.70 * len(unique_runs))
    n_val = int(0.15 * len(unique_runs))
    test_runs = unique_runs[n_train + n_val:]
    mask = np.isin(run_ids, test_runs)

    X_test = X[mask].copy()
    y_test = y[mask]

    # Same preprocessing as training.
    X_test[:, :, 0] /= 10.0

    checkpoint = torch.load(
        ROOT / "artifacts/model.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = DriftRegressor(
        window_size=checkpoint["window_size"],
        channels=checkpoint["channels"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        prediction = model(
            torch.from_numpy(X_test)
        ).numpy()

    report_metrics("Neural network", y_test, prediction)

    # Transparent simple baseline using the last observed inputs.
    raw = X[mask]
    baseline = (
        0.010 * raw[:, -1, 0]
        + 0.004 * raw[:, -1, 1]
    )
    report_metrics("Simple proxy baseline", y_test, baseline)


if __name__ == "__main__":
    main()
