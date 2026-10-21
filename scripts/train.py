import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ai.model import DriftRegressor
from edge_ai.experiment import (
    environment, load_dataset, scale_inputs, sha256, split_runs, write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_dataset.npz")
    parser.add_argument("--output", default="artifacts/model.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("epochs and threads must be positive")
    torch.set_num_threads(args.threads)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = ROOT / args.data
    X, y, run_ids, dataset_metadata = load_dataset(data_path)

    # Split independent runs, not individual windows.
    split = split_runs(run_ids, args.seed)
    train_runs = split["train"]
    val_runs = split["validation"]

    train_mask = np.isin(run_ids, train_runs)
    val_mask = np.isin(run_ids, val_runs)

    if not train_mask.any() or not val_mask.any():
        raise ValueError("Not enough runs for train/validation split")

    # Fixed input scaling; same convention will be used in C++.
    X = scale_inputs(X)
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
    provenance = {
        "schema_version": 1, "dataset_sha256": sha256(data_path),
        "dataset_metadata": dataset_metadata, "split": split,
        "preprocessing": {"temperature_divisor": 10.0, "vibration_divisor": 1.0},
        "training": {"seed": args.seed, "epochs": args.epochs,
                     "threads": args.threads, "batch_size": 128,
                     "optimizer": "Adam", "learning_rate": 0.001,
                     "loss": "MSE", "device": "cpu"},
        "environment": environment(),
    }
    history = []

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
        history.append({"epoch": epoch + 1, "validation_rmse_mm": val_loss ** 0.5})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    **provenance,
                    "best_epoch": epoch + 1,
                    "best_validation_rmse_mm": val_loss ** 0.5,
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
    write_json(output.with_suffix(".training.json"), {
        **provenance, "model_sha256": sha256(output),
        "best_validation_rmse_mm": best_val_loss ** 0.5, "history": history,
    })


if __name__ == "__main__":
    main()
