import re
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts" / "model.onnx"
EXE = ROOT / "build" / "Release" / "edge_ai_cpp.exe"

def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"ONNX model not found: {MODEL}")
    if not EXE.exists():
        raise FileNotFoundError(
            f"C++ executable not found: {EXE}. Build the project first."
        )

    # Reproduce the exact synthetic input generated in cpp/main.cpp.
    x = np.zeros((1, 32, 2), dtype=np.float32)

    for t in range(32):
        temperature_delta = np.float32(
            1.0 + 2.0 * t / 31.0
        )
        vibration = np.float32(
            0.15 + 0.10 * (t % 8) / 7.0
        )

        x[0, t, 0] = temperature_delta / 10.0
        x[0, t, 1] = vibration

    # Python ONNX Runtime prediction
    session = ort.InferenceSession(
        str(MODEL),
        providers=["CPUExecutionProvider"],
    )
    python_prediction = float(
        session.run(["predicted_drift_mm"], {
            "sensor_window": x
        })[0][0]
    )

    # Run the C++ program using the same embedded input.
    result = subprocess.run(
        [str(EXE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    match = re.search(
        r"Predicted drift:\s*([-+]?\d+(?:\.\d+)?)\s*mm",
        result.stdout,
    )
    if not match:
        print(result.stdout)
        raise RuntimeError("Could not parse C++ prediction.")

    cpp_prediction = float(match.group(1))
    difference = abs(python_prediction - cpp_prediction)
    tolerance = 1e-5  # mm; C++ output is rounded to six decimals.

    print("Python ONNX prediction :", f"{python_prediction:.9f}", "mm")
    print("C++ ONNX prediction    :", f"{cpp_prediction:.9f}", "mm")
    print("Absolute difference    :", f"{difference:.9f}", "mm")
    print("Allowed tolerance      :", f"{tolerance:.9f}", "mm")

    if difference <= tolerance:
        print("\nPASS: Python and C++ predictions match within tolerance.")
    else:
        raise SystemExit(
            "\nFAIL: Predictions differ beyond tolerance."
        )

if __name__ == "__main__":
    main()
