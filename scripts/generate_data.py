import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ai.dataset import make_runs, windows_from_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_dataset.npz")
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runs = make_runs(
        num_runs=args.runs,
        steps_per_run=args.steps,
        seed=args.seed,
    )
    X, y, run_ids = windows_from_runs(runs)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generator": "make_runs_v1", "seed": args.seed,
        "num_runs": args.runs, "steps_per_run": args.steps,
        "sample_period_s": 0.1, "window_size": 32, "forecast_steps": 20,
        "temperature_reference": "first measurement of each run",
        "vibration_units": "arbitrary synthetic units",
    }
    np.savez_compressed(output, X=X, y=y, run_ids=run_ids,
                        metadata=json.dumps(metadata, sort_keys=True))

    print(f"Saved dataset: {output}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Independent runs: {len(np.unique(run_ids))}")


if __name__ == "__main__":
    main()
