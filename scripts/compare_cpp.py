"""Compare multiple preprocessed windows through Python and C++ ONNX Runtime."""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from edge_ai.experiment import environment, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/cpp_parity.json")
    args = parser.parse_args()
    model = ROOT / "artifacts/model.onnx"
    exe = ROOT / "build/Release/edge_ai_cpp.exe"
    if not exe.exists():
        raise FileNotFoundError("Build the Release C++ demo first.")
    rng = np.random.default_rng(123)
    windows = rng.uniform([0.0, 0.0], [0.4, 0.5], size=(16, 32, 2)).astype(np.float32)
    windows[0] = 0
    for t in range(32):
        windows[1, t] = [(1.0 + 2.0 * t / 31.0) / 10.0, 0.15 + 0.10 * (t % 8) / 7.0]
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    records = []
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        path = Path(temp) / "window.txt"
        for index, window in enumerate(windows):
            np.savetxt(path, window.reshape(-1), fmt="%.9g")
            result = subprocess.run(
                [str(exe), "--input", str(path), "--predict-only"],
                cwd=ROOT, text=True, capture_output=True, check=True,
            )
            match = re.search(r"Predicted drift:\s*([-+0-9.eE]+)\s*mm", result.stdout)
            if not match:
                raise RuntimeError("Could not parse C++ prediction: " + result.stdout)
            cpp = float(match.group(1))
            python = float(session.run(["predicted_drift_mm"], {"sensor_window": window[None]})[0][0])
            difference = abs(python - cpp)
            records.append({"case": index, "python_mm": python, "cpp_mm": cpp,
                            "absolute_difference_mm": difference,
                            "passed": bool(np.isfinite(difference) and difference <= 1e-6)})
    passed = all(row["passed"] for row in records)
    write_json(ROOT / args.output, {
        "environment": environment(), "model_sha256": sha256(model),
        "executable_sha256": sha256(exe), "seed": 123, "tolerance_mm": 1e-6,
        "scope": "zero window, demo window, 14 random preprocessed windows; no accuracy claim",
        "passed": passed, "cases": records,
    })
    print(f"{'PASS' if passed else 'FAIL'}: {len(records)} Python/C++ cases")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
