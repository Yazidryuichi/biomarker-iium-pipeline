"""Stage 5 unit tests: DeLong test, bootstrap CIs, LOSO machinery, model factory.

These tests exercise the statistical machinery of `stage5_fair_comparison.py`
without running the full pipeline. They use a small mocked dataset (N=30
subjects, 12 features) where the ground-truth signal is engineered into the
features, so the classifiers should clear chance with high power. CI also
runs the full pipeline against the synthetic fixture, but that path is gated
behind SKIP_STAGE5=1 because Stage 5 is statistically expensive (100-fold CV
× 1000-permutation × 10000-bootstrap). These unit tests cover the math.
"""

import os
import sys

import numpy as np
import pytest

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_ROOT)

from stages.stage5_fair_comparison import (
    N_BOOT,
    N_PERM,
    N_SELECT,
    N_SPLITS,
    _delong_paired_p,
    _evaluate_loso,
    _evaluate_per_fold,
    _make_pipeline,
    _model_factory,
    _permutation_bacc,
    _resolve_stage5_params,
    _run_cell,
    _subject_bootstrap_bacc_ci,
    _subject_bootstrap_ci,
)


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


def _make_mock_dataset(n_subjects: int = 30, n_features: int = 12,
                       signal_strength: float = 1.0, seed: int = 0):
    """Construct a small mocked dataset with a known classification signal.

    Returns (X, y) where:
      - X is shape (n_subjects, n_features), gaussian noise + a class-shifted
        signal in the first 3 features
      - y is balanced binary labels
      - signal_strength controls separability (1.0 = easy, 0.0 = pure noise)
    """
    rng = np.random.default_rng(seed)
    half = n_subjects // 2
    y = np.array([0] * half + [1] * (n_subjects - half), dtype=int)
    X = rng.standard_normal((n_subjects, n_features))
    # Inject signal: first 3 features carry class-shifted mean
    X[y == 1, :3] += signal_strength
    return X, y


# ──────────────────────────────────────────────────────────────────
# Pipeline + model factory smoke tests
# ──────────────────────────────────────────────────────────────────


def test_model_factory_known_kinds():
    """All documented model kinds resolve to callable factories."""
    for kind in ("svm_linear", "rf_shallow", "svm_l1", "lr_l1", "lr_elasticnet"):
        factory = _model_factory(kind)
        model = factory()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


def test_model_factory_unknown_raises():
    with pytest.raises(ValueError, match="unknown model kind"):
        _model_factory("nonexistent_model")


def test_make_pipeline_structure():
    """Pipeline has the expected 3-step structure: scaler -> select -> clf."""
    pipe = _make_pipeline(_model_factory("svm_linear")())
    step_names = [s[0] for s in pipe.steps]
    assert step_names == ["scaler", "select", "clf"]
    # SelectKBest k matches the module constant
    assert pipe.named_steps["select"].k == N_SELECT


# ──────────────────────────────────────────────────────────────────
# Per-fold + LOSO evaluators
# ──────────────────────────────────────────────────────────────────


def test_per_fold_returns_correct_shape():
    """_evaluate_per_fold returns one BAcc + one AUC per fold.

    Splits must be stratified — `_make_mock_dataset` lays out class 0 in
    indices [0, n/2) and class 1 in [n/2, n), so train sets that include
    ONLY low or ONLY high indices contain a single class and SVC errors.
    """
    X, y = _make_mock_dataset(n_subjects=20)
    # Stratified 2-fold: each fold has 5 of each class in train + test
    splits = [
        (np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14]),
         np.array([5, 6, 7, 8, 9, 15, 16, 17, 18, 19])),
        (np.array([5, 6, 7, 8, 9, 15, 16, 17, 18, 19]),
         np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14])),
    ]
    factory = _model_factory("svm_linear")
    accs, aucs = _evaluate_per_fold(X, y, factory, splits)
    assert accs.shape == (2,)
    assert aucs.shape == (2,)
    # BAcc is in [0, 1]; AUC is in [0, 1] (or NaN if degenerate)
    assert np.all((accs >= 0) & (accs <= 1))
    valid = ~np.isnan(aucs)
    assert np.all((aucs[valid] >= 0) & (aucs[valid] <= 1))


def test_per_fold_recovers_signal_at_high_strength():
    """With strong signal, BAcc should clear 0.6 mean across folds."""
    X, y = _make_mock_dataset(n_subjects=30, signal_strength=2.0)
    splits = [(np.arange(0, 20), np.arange(20, 30)),
              (np.arange(10, 30), np.arange(0, 10))]
    factory = _model_factory("svm_linear")
    accs, _ = _evaluate_per_fold(X, y, factory, splits)
    assert accs.mean() > 0.6, f"Expected BAcc > 0.6, got {accs.mean():.3f}"


def test_loso_shapes_and_ranges():
    """LOSO returns one proba/pred per subject; probas in [0, 1]."""
    X, y = _make_mock_dataset(n_subjects=15)
    factory = _model_factory("svm_linear")
    proba, label, pred = _evaluate_loso(X, y, factory)
    assert proba.shape == (15,)
    assert label.shape == (15,)
    assert pred.shape == (15,)
    valid = ~np.isnan(proba)
    assert np.all((proba[valid] >= 0) & (proba[valid] <= 1))
    assert set(np.unique(pred)).issubset({0, 1})
    np.testing.assert_array_equal(label, y)


# ──────────────────────────────────────────────────────────────────
# Statistical inference: DeLong + bootstrap + permutation
# ──────────────────────────────────────────────────────────────────


def test_delong_paired_p_returns_valid_range():
    """DeLong test returns AUCs in [0,1], z-score finite, p in [0,1]."""
    rng = np.random.default_rng(42)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    proba1 = rng.uniform(0, 1, size=10)
    proba2 = rng.uniform(0, 1, size=10)
    auc1, auc2, z, p = _delong_paired_p(proba1, proba2, y)
    assert 0 <= auc1 <= 1
    assert 0 <= auc2 <= 1
    assert np.isfinite(z)
    assert 0 <= p <= 1


def test_delong_identical_classifiers_high_p():
    """Two identical classifiers should produce p ≈ 1 (no difference)."""
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.7, 0.8, 0.9, 0.85, 0.95])
    auc1, auc2, z, p = _delong_paired_p(proba, proba.copy(), y)
    assert auc1 == auc2
    # Identical => z=0 NaN division yields p that should be high (≥0.9)
    # In practice _delong_paired_p may return p=1.0 or p=NaN for this edge
    assert np.isnan(p) or p >= 0.9, f"Expected p≈1, got {p}"


def test_subject_bootstrap_ci_bounds():
    """Subject-bootstrap CI returns (point_est, lo, hi) with lo <= point <= hi.

    Uses overlapping proba ranges (rather than perfectly separated) so
    bootstrap resampling produces AUC variance — perfect separation gives
    AUC=1.0 every resample and the CI collapses to zero width, which is
    correct but not informative as a sanity check.
    """
    rng = np.random.default_rng(0)
    y = np.array([0] * 10 + [1] * 10)
    # Good-but-not-perfect classifier: overlapping probas
    proba = np.where(y == 1, rng.uniform(0.3, 0.9, size=20),
                     rng.uniform(0.1, 0.7, size=20))
    auc, lo, hi = _subject_bootstrap_ci(proba, y, n_boot=200)
    assert 0 <= lo <= auc <= hi <= 1, (
        f"Bootstrap CI ordering violated: lo={lo}, auc={auc}, hi={hi}"
    )
    # Width should be non-degenerate at this overlap level
    assert hi - lo > 0.01, f"CI width too tight (lo={lo}, hi={hi}); test " \
                            "fixture's overlap may be too small"


def test_subject_bootstrap_bacc_ci_bounds():
    """Same ordering check for BAcc bootstrap."""
    rng = np.random.default_rng(1)
    y = np.array([0] * 10 + [1] * 10)
    pred = np.where(rng.uniform(0, 1, size=20) > 0.5, 1, 0)
    bacc, lo, hi = _subject_bootstrap_bacc_ci(pred, y, n_boot=200)
    assert 0 <= lo <= bacc <= hi <= 1


def test_permutation_test_returns_pvalue_in_range():
    """Permutation test returns observed BAcc + p-value in [0, 1]."""
    X, y = _make_mock_dataset(n_subjects=20, signal_strength=0.0)  # pure noise
    splits = [(np.arange(0, 14), np.arange(14, 20)),
              (np.arange(6, 20), np.arange(0, 6))]
    factory = _model_factory("svm_linear")
    # Small n_perm for speed; we're testing the API not the statistical power
    obs, p = _permutation_bacc(X, y, factory, splits, n_perm=20)
    assert 0 <= p <= 1
    assert 0 <= obs <= 1


# ──────────────────────────────────────────────────────────────────
# Run-cell integration
# ──────────────────────────────────────────────────────────────────


def test_run_cell_smoke():
    """_run_cell completes end-to-end and produces all expected fields."""
    X, y = _make_mock_dataset(n_subjects=20)
    # Stratified 2-fold splits
    splits = [(np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14]),
               np.array([5, 6, 7, 8, 9, 15, 16, 17, 18, 19])),
              (np.array([5, 6, 7, 8, 9, 15, 16, 17, 18, 19]),
               np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14]))]
    result = _run_cell("test_cell", "test_fset", "svm_linear", X, y, splits)
    assert result.name == "test_cell"
    assert result.feature_set == "test_fset"
    assert result.model_class == "svm_linear"
    assert result.per_fold_bacc.shape == (2,)
    assert result.per_fold_auc.shape == (2,)
    assert result.per_subject_proba.shape == (20,)
    assert result.per_subject_label.shape == (20,)
    assert result.per_subject_pred.shape == (20,)


# ──────────────────────────────────────────────────────────────────
# Config resolution
# ──────────────────────────────────────────────────────────────────


def test_resolve_params_module_defaults_when_no_config():
    """No config + no CI_FAST + no CLI override => module defaults."""
    # Ensure CI_FAST is not set during this test
    old_ci_fast = os.environ.pop("CI_FAST", None)
    try:
        params, cfg = _resolve_stage5_params(None, None)
        assert params["n_splits"] == N_SPLITS
        assert params["n_repeats"] == 10  # N_REPEATS
        assert params["n_perm"] == N_PERM
        assert cfg == {}
    finally:
        if old_ci_fast is not None:
            os.environ["CI_FAST"] = old_ci_fast


def test_resolve_params_cli_override_wins():
    """CLI --n-perm takes priority over everything else."""
    params, _ = _resolve_stage5_params(None, cli_n_perm=42)
    assert params["n_perm"] == 42


def test_resolve_params_ci_fast_env(monkeypatch):
    """CI_FAST=1 reduces n_perm without a config file."""
    monkeypatch.setenv("CI_FAST", "1")
    params, _ = _resolve_stage5_params(None, None)
    assert params["n_perm"] == 10  # default fallback when no config provides ci_fast_n_perm
