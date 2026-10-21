"""Physics-inspired synthetic sensor data. Not calibrated to real machinery."""

import numpy as np


def make_runs(
    num_runs=60,
    steps_per_run=200,
    sample_period_s=0.1,
    seed=42,
):
    """Create independent simulated machine runs."""
    if num_runs < 3:
        raise ValueError("num_runs must be at least 3")
    if steps_per_run < 10:
        raise ValueError("steps_per_run must be at least 10")
    if sample_period_s <= 0:
        raise ValueError("sample_period_s must be positive")

    rng = np.random.default_rng(seed)
    time = np.arange(steps_per_run) * sample_period_s
    runs = []

    for _ in range(num_runs):
        # Simplified warm-up curve
        ambient = rng.uniform(18.0, 25.0)
        rise = rng.uniform(1.0, 8.0)
        tau = rng.uniform(25.0, 100.0)

        temperature = ambient + rise * (
            1.0 - np.exp(-time / tau)
        )
        temperature += 0.15 * np.sin(
            2 * np.pi * time / rng.uniform(18.0, 35.0)
        )
        temperature += rng.normal(
            0.0, 0.035, steps_per_run
        )

        # Synthetic vibration signal
        vibration = rng.uniform(0.05, 0.4)
        vibration = vibration + 0.07 * np.sin(
            2 * np.pi * time / rng.uniform(1.5, 4.0)
        )
        vibration += rng.normal(
            0.0, 0.025, steps_per_run
        )
        vibration = np.clip(vibration, 0.0, None)

        temperature_delta = temperature - temperature[0]

        # Illustrative target equation, not a real machine model.
        drift_mm = (
            0.010 * temperature_delta
            + 0.004 * vibration
            + 0.00035
            * np.maximum(temperature_delta - 3.0, 0.0) ** 2
        )
        drift_mm += rng.normal(
            0.0, 0.00008, steps_per_run
        )

        runs.append({
            "temperature_delta_c": temperature_delta.astype(
                np.float32
            ),
            "vibration_rms": vibration.astype(np.float32),
            "drift_mm": drift_mm.astype(np.float32),
        })

    return runs


def windows_from_runs(runs, window_size=32, forecast_steps=20):
    """Create windows without crossing run boundaries.

    Returns:
        X: sensor windows, shape [samples, time, 2]
        y: future drift in mm, shape [samples]
        run_ids: originating run for each sample
    """
    if window_size < 2 or forecast_steps < 0:
        raise ValueError("Invalid window or forecast size")

    features, targets, run_ids = [], [], []

    for run_id, run in enumerate(runs):
        length = len(run["drift_mm"])
        last_start = length - window_size - forecast_steps

        for start in range(last_start + 1):
            end = start + window_size
            target_index = end - 1 + forecast_steps

            window = np.column_stack((
                run["temperature_delta_c"][start:end],
                run["vibration_rms"][start:end],
            ))

            features.append(window)
            targets.append(run["drift_mm"][target_index])
            run_ids.append(run_id)

    if not features:
        raise ValueError("No windows generated")

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(run_ids, dtype=np.int32),
    )
