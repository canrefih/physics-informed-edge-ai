import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ai.model import DriftRegressor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_dataset.npz")
    parser.add_argument("--output", default="artifacts/model.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = np.load(ROOT / args.data)
    X = data["X"].copy()
    y = data["y"]
    run_ids = data["run_ids"]

    # Split independent runs, not individual windows.
    unique_runs = np.unique(run_ids)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_runs)

    n_train = int(0.70 * len(unique_runs))
    n_val = int(0.15 * len(unique_runs))

    train_runs = unique_runs[:n_train]
    val_runs = unique_runs[n_train:n_train + n_val]

    train_mask = np.isin(run_ids, train_runs)
    val_mask = np.isin(run_ids, val_runs)

    if not train_mask.any() or not val_mask.any():
        raise ValueError("Not enough runs for train/validation split")

    # Fixed input scaling; same convention will be used in C++.
    X[:, :, 0] /= 10.0
    # Vibration is already expressed in the generator's chosen units.

    model = DriftRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(X[train_mask]),
        torch.from_numpy(y[train_mask]),
    )
    loader = DataLoader(
        train_ds, batch_size=128, shuffle=True
    )

    val_x = torch.from_numpy(X[val_mask])
    val_y = torch.from_numpy(y[val_mask])

    best_val_loss = float("inf")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()

        for xb, yb in loader:
            optimizer.zero_grad()
            predictions = model(xb)
            loss = loss_fn(predictions, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x), val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "window_size": 32,
                    "channels": 2,
                    "hidden_dim": 64,
                },
                output,
            )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:03d}/{args.epochs} | "
                f"Validation RMSE: {val_loss**0.5:.6f} mm"
            )

    print(f"Best validation RMSE: {best_val_loss**0.5:.6f} mm")
    print(f"Saved model: {output}")


if __name__ == "__main__":
    main()
