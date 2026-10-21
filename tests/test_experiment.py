"""Regression checks for checkpoint-bound evaluation and split isolation."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from edge_ai.experiment import sha256, split_runs, validate_checkpoint, validate_split

ROOT = Path(__file__).resolve().parents[1]


def test_split_is_disjoint_and_nondefault_seed_is_preserved():
    ids = np.repeat(np.arange(60), 4)
    split = split_runs(ids, 7)
    validate_split(split, ids)
    assert split != split_runs(ids, 42)
    assert [len(split[name]) for name in ("train", "validation", "test")] == [42, 9, 9]
    split["test"][0] = split["train"][0]
    with pytest.raises(ValueError, match="overlap"):
        validate_split(split, ids)


def test_legacy_checkpoint_and_changed_data_are_rejected(tmp_path):
    path = tmp_path / "data.npz"
    path.write_bytes(b"original")
    ids = np.arange(10)
    with pytest.raises(ValueError, match="Legacy"):
        validate_checkpoint({}, path, ids)
    checkpoint = {
        "schema_version": 1, "dataset_sha256": sha256(path),
        "split": split_runs(ids, 7),
        "preprocessing": {"temperature_divisor": 10.0, "vibration_divisor": 1.0},
    }
    validate_checkpoint(checkpoint, path, ids)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        validate_checkpoint(checkpoint, path, ids)


def test_evaluation_uses_saved_split_for_nondefault_training_seed(tmp_path):
    data, model, report = (tmp_path / name for name in ("data.npz", "model.pt", "evaluation.json"))
    commands = [
        ["generate_data.py", "--runs", "10", "--steps", "70", "--output", str(data)],
        ["train.py", "--data", str(data), "--output", str(model), "--seed", "7", "--epochs", "2"],
        ["evaluate.py", "--data", str(data), "--model", str(model), "--output", str(report)],
    ]
    for script, *args in commands:
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                       cwd=ROOT, capture_output=True, text=True, check=True)
    checkpoint = torch.load(model, weights_only=True)
    evaluation = json.loads(report.read_text())
    assert evaluation["split"] == checkpoint["split"] == split_runs(np.arange(10), 7)
    assert evaluation["model_sha256"] == sha256(model)
    assert evaluation["sample_counts"]["test"] == 2 * (70 - 32 - 20 + 1)
    assert set(evaluation["metrics"]) == {
        "neural", "last_value_proxy", "linear_full_window", "ridge_full_window",
    }
