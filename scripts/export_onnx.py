import sys
from pathlib import Path

import torch
import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ai.model import DriftRegressor
from edge_ai.experiment import environment, sha256, write_json


def main():
    checkpoint_path = ROOT / "artifacts/model.pt"
    output_path = ROOT / "artifacts/model.onnx"

    checkpoint = torch.load(
        checkpoint_path,
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

    example_input = torch.zeros(
        1, checkpoint["window_size"], checkpoint["channels"]
    )

    torch.onnx.export(
        model,
        (example_input,),
        str(output_path),
        input_names=["sensor_window"],
        output_names=["predicted_drift_mm"],
        dynamic_axes={
            "sensor_window": {0: "batch_size"},
            "predicted_drift_mm": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )

    exported_model = onnx.load(str(output_path))
    onnx.checker.check_model(exported_model)
    write_json(output_path.with_suffix(".export.json"), {
        "environment": environment(), "checkpoint_sha256": sha256(checkpoint_path),
        "onnx_sha256": sha256(output_path), "opset": 17,
        "input_name": "sensor_window", "output_name": "predicted_drift_mm",
        "input_shape": ["batch", 32, 2], "dtype": "float32",
        "preprocessing": {"temperature_divisor": 10.0, "vibration_divisor": 1.0},
    })

    print(f"ONNX model validated: {output_path}")


if __name__ == "__main__":
    main()
