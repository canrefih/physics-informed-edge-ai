"""Record repeated C++ benchmark runs and their environment."""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from edge_ai.experiment import environment, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default="results/cpp_benchmark.json")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    exe = ROOT / "build/Release/edge_ai_cpp.exe"
    records = []
    for _ in range(args.runs):
        result = subprocess.run([str(exe)], cwd=ROOT, capture_output=True, text=True, check=True)
        stats = {}
        for name in ("Average", "Median", "P95", "P99", "Min", "Max"):
            match = re.search(name + r" latency:\s*([0-9.]+) us", result.stdout)
            if not match:
                raise RuntimeError("Missing benchmark field: " + name)
            stats[name.lower() + "_us"] = float(match.group(1))
        records.append({"metrics": stats, "stdout": result.stdout})
    report = {
        "environment": environment(), "cpu_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
        "model_sha256": sha256(ROOT / "artifacts/model.onnx"),
        "executable_sha256": sha256(exe), "build": "Release",
        "cpp_onnxruntime_version": (ROOT / "third_party/onnxruntime-win-x64-1.31.0/VERSION_NUMBER").read_text().strip(),
        "intra_op_threads": 1, "warmup_iterations_per_run": 100,
        "measured_iterations_per_run": 5000,
        "scope": "steady_clock around Session::Run on the same preallocated CPU input; output allocation included",
        "excluded": ["sensor acquisition", "preprocessing", "model loading", "startup", "communication", "control response"],
        "runs": records,
    }
    write_json(ROOT / args.output, report)
    print(f"Saved {len(records)} benchmark runs: {ROOT / args.output}")


if __name__ == "__main__":
    main()
