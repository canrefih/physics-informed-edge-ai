import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_ai.dataset import make_runs, windows_from_runs


def test_output_shapes_and_finite_values():
    runs = make_runs(4, 80, seed=7)
    X, y, ids = windows_from_runs(
        runs, window_size=10, forecast_steps=5
    )

    assert X.shape[1:] == (10, 2)
    assert y.shape == (len(X),)
    assert ids.shape == (len(X),)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


def test_windows_stay_inside_each_run():
    runs = make_runs(3, 70, seed=11)
    X, y, ids = windows_from_runs(
        runs, window_size=8, forecast_steps=2
    )

    assert len(X) == 3 * (70 - 8 - 2 + 1)
    assert set(ids) == {0, 1, 2}
    windows_per_run = 70 - 8 - 2 + 1
    for index, (window, target, run_id) in enumerate(zip(X, y, ids)):
        start = index % windows_per_run
        run = runs[run_id]
        np.testing.assert_array_equal(window[:, 0], run["temperature_delta_c"][start:start + 8])
        np.testing.assert_array_equal(window[:, 1], run["vibration_rms"][start:start + 8])
        assert target == run["drift_mm"][start + 8 - 1 + 2]


def test_invalid_run_count():
    with pytest.raises(ValueError):
        make_runs(num_runs=2)


def test_impossible_window():
    runs = make_runs(3, 30, seed=5)
    with pytest.raises(ValueError):
        windows_from_runs(
            runs, window_size=40, forecast_steps=5
        )
