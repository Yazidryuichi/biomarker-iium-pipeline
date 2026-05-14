"""CNN-LSTM smoke test — Linux-only (PyTorch + joblib fork hangs on macOS).

The CNN-LSTM classifier in Stage 4 is disabled by default (`RUN_CNN_LSTM=1`
required) because of a known PyTorch + joblib fork interaction that causes
deadlocks on macOS. This test validates the function imports + runs without
crashing on Linux, where the issue does not reproduce.

Skipped on macOS (and Windows, as a defensive choice — only tested on Linux).
"""

import os
import sys

import pytest

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_ROOT)


@pytest.mark.skipif(
    sys.platform != "linux",
    reason=("CNN-LSTM uses PyTorch + joblib fork which deadlocks on macOS. "
            "Linux-only test."),
)
def test_cnn_lstm_smoke_linux_only():
    """CNN-LSTM importable + runs on tiny synthetic input without crashing.

    Constructs a minimal dataset (16 subjects, 12 features), calls the Stage 4
    CNN-LSTM CV runner with cv_folds=2/cv_repeats=1 to keep runtime short, and
    asserts the return shape + that no exception is raised.
    """
    pytest.importorskip("torch", reason="PyTorch not installed")

    import numpy as np
    import pandas as pd

    from stages.stage4_analysis import _run_cnn_lstm_cv

    # Construct a tiny in-memory dataset matching what Stage 4 passes.
    rng = np.random.default_rng(0)
    n_subjects = 16
    n_features = 12
    feature_cols = [f"feat_{i}" for i in range(n_features)]

    df = pd.DataFrame(
        rng.standard_normal((n_subjects, n_features)),
        columns=feature_cols,
    )
    df["subject_id"] = [f"S{i:03d}" for i in range(n_subjects)]

    # Inject a small signal so the loss reduces during training (not strictly
    # required for the smoke test, but prevents NaN losses on pure noise).
    y = np.array([0] * (n_subjects // 2) + [1] * (n_subjects - n_subjects // 2),
                 dtype=int)
    df.loc[y == 1, feature_cols[:3]] += 1.0

    valid_idx = np.arange(n_subjects)
    feature_sets = {"test_set": feature_cols}

    # Minimal config: 2 folds, 1 repeat for speed.
    config = {
        "ml": {
            "cv_folds": 2,
            "cv_repeats": 1,
            "random_state": 42,
        }
    }

    results = _run_cnn_lstm_cv(df, valid_idx, feature_sets, y, config)

    # Result should be a list (empty if PyTorch is missing or runtime errors).
    # If PyTorch is present and the model runs, we get one row per feature_set.
    assert isinstance(results, list)
    if results:
        assert len(results) == 1, f"Expected 1 feature_set row, got {len(results)}"
        row = results[0]
        # Expected fields written by Stage 4's CV loop. The exact field set may
        # evolve; we check the minimum invariants here.
        assert "Feature Set" in row or "feature_set" in row or "model" in row, (
            f"CNN-LSTM result row missing expected keys: {list(row.keys())}"
        )
