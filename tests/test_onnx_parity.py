from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from edge_ai.model import DriftRegressor


ROOT = Path(__file__).resolve().parents[1]


def test_onnx_matches_pytorch():
    checkpoint = torch.load(
        ROOT / "artifacts/model.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = DriftRegressor(
        checkpoint["window_size"],
        checkpoint["channels"],
        checkpoint["hidden_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    rng = np.random.default_rng(123)
    sample = rng.normal(
        0, 0.1,
        (8, checkpoint["window_size"], checkpoint["channels"]),
    ).astype(np.float32)

    with torch.no_grad():
        expected = model(torch.from_numpy(sample)).numpy()

    session = ort.InferenceSession(
        str(ROOT / "artifacts/model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(
        ["predicted_drift_mm"],
        {"sensor_window": sample},
    )[0]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-6,
    )
