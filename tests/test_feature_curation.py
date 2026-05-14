"""
Tests for stages/analysis/feature_curation.py.

Synthetic-data tests for:
  - drop_low_variance: zero-variance / near-constant columns dropped.
  - drop_collinear_hierarchical: highly correlated pairs collapse to a
    single representative (highest-variance member).
  - cluster_map structure: kept feature -> list of dropped followers.
  - Edge cases: all-constant matrix, single-feature matrix.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_ROOT)

from stages.analysis.feature_curation import (  # noqa: E402
    drop_low_variance, drop_collinear_hierarchical,
)


# ---------------------------------------------------------------- drop_low_variance

def test_drop_low_variance_drops_zero_variance_columns():
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame({
        "useful":  rng.standard_normal(n),
        "constant": np.ones(n) * 3.0,
        "zero":     np.zeros(n),
        "noisy":    rng.standard_normal(n),
    })
    out = drop_low_variance(X, threshold=1e-6)
    kept = list(out["X_filtered"].columns)
    assert "useful" in kept
    assert "noisy" in kept
    assert "constant" not in kept
    assert "zero" not in kept
    assert out["X_filtered"].shape == (n, 2)


def test_drop_low_variance_threshold_respected():
    rng = np.random.default_rng(1)
    n = 50
    X = pd.DataFrame({
        "high_var": rng.standard_normal(n) * 5.0,
        "low_var":  rng.standard_normal(n) * 1e-4,
    })
    # variance of low_var ~ 1e-8, of high_var ~ 25
    out = drop_low_variance(X, threshold=1e-6)
    assert "high_var" in out["X_filtered"].columns
    assert "low_var" not in out["X_filtered"].columns


def test_drop_low_variance_all_constant_drops_everything():
    n = 20
    X = pd.DataFrame({"a": np.ones(n), "b": np.full(n, 2.0)})
    out = drop_low_variance(X, threshold=1e-6)
    assert out["X_filtered"].shape == (n, 0)
    assert len(out["dropped_idx"]) == 2


def test_drop_low_variance_single_feature():
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"only": rng.standard_normal(30)})
    out = drop_low_variance(X, threshold=1e-6)
    assert out["X_filtered"].shape == (30, 1)
    assert list(out["X_filtered"].columns) == ["only"]


# ---------------------------------------------------------- drop_collinear_hierarchical

def test_drop_collinear_collapses_perfectly_correlated_pair():
    rng = np.random.default_rng(3)
    n = 60
    base = rng.standard_normal(n)
    X = pd.DataFrame({
        # An almost-perfectly correlated pair — one should drop.
        "x_high_var":  base * 4.0 + rng.standard_normal(n) * 0.001,
        "x_low_var":   base * 0.5 + rng.standard_normal(n) * 0.001,
        # Independent feature — should remain.
        "indep":       rng.standard_normal(n),
    })
    out = drop_collinear_hierarchical(X, list(X.columns), corr_threshold=0.95)
    kept = out["kept_names"]
    # Highest-variance representative of the correlated pair survives:
    assert "x_high_var" in kept
    assert "x_low_var" not in kept
    assert "indep" in kept


def test_drop_collinear_cluster_map_records_dropped_followers():
    rng = np.random.default_rng(4)
    n = 80
    base_a = rng.standard_normal(n)
    base_b = rng.standard_normal(n)
    X = pd.DataFrame({
        "a1": base_a * 5.0 + rng.standard_normal(n) * 0.001,
        "a2": base_a * 1.0 + rng.standard_normal(n) * 0.001,
        "a3": base_a * 0.3 + rng.standard_normal(n) * 0.001,
        "b1": base_b * 4.0 + rng.standard_normal(n) * 0.001,
        "b2": base_b * 0.8 + rng.standard_normal(n) * 0.001,
    })
    out = drop_collinear_hierarchical(X, list(X.columns), corr_threshold=0.95)

    # Two clusters, two kept features.
    assert len(out["kept_names"]) == 2
    cluster_map = out["cluster_map"]
    # Every kept name is a key in cluster_map.
    for kept in out["kept_names"]:
        assert kept in cluster_map
    # Followers must not appear as kept names; every non-kept feature must
    # appear in exactly one follower list.
    all_followers = [f for fs in cluster_map.values() for f in fs]
    assert set(all_followers) == set(out["dropped_names"])
    assert len(all_followers) == len(set(all_followers))  # no duplicates


def test_drop_collinear_keeps_uncorrelated_features():
    rng = np.random.default_rng(5)
    n = 100
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(5)})
    out = drop_collinear_hierarchical(X, list(X.columns), corr_threshold=0.95)
    # Random independent features should not collapse with N=100.
    assert len(out["kept_names"]) == 5
    assert out["dropped_names"] == []


def test_drop_collinear_single_feature_passthrough():
    rng = np.random.default_rng(6)
    X = pd.DataFrame({"only": rng.standard_normal(20)})
    out = drop_collinear_hierarchical(X, ["only"], corr_threshold=0.95)
    assert out["kept_names"] == ["only"]
    assert out["dropped_names"] == []
    assert out["cluster_map"] == {"only": []}


def test_drop_collinear_zero_features():
    X = pd.DataFrame(np.zeros((10, 0)))
    out = drop_collinear_hierarchical(X, [], corr_threshold=0.95)
    assert out["kept_names"] == []
    assert out["dropped_names"] == []


def test_drop_collinear_numpy_array_input():
    rng = np.random.default_rng(7)
    n = 50
    base = rng.standard_normal(n)
    arr = np.column_stack([
        base * 3.0,
        base + rng.standard_normal(n) * 0.001,
        rng.standard_normal(n),
    ])
    names = ["a", "b", "c"]
    out = drop_collinear_hierarchical(arr, names, corr_threshold=0.95)
    # Should still work on raw ndarray input; one of {a, b} dropped.
    assert "c" in out["kept_names"]
    assert len(out["kept_names"]) == 2
