"""
Tests for stages/analysis/target_preparation.py.

  - residualize_target: synthetic data with a known age effect should
    yield residuals near zero plus the injected noise; coefficients
    should recover the planted slope.
  - derive_clinical_threshold: each method produces the expected
    prevalence (within +/- 1 sample at small N) and the at-risk
    direction is "lower residual = at risk".
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_ROOT)

from stages.analysis.target_preparation import (  # noqa: E402
    residualize_target, derive_clinical_threshold,
)


# ------------------------------------------------------------ residualize_target

def test_residualize_target_recovers_known_slope():
    rng = np.random.default_rng(0)
    n = 200
    age = rng.uniform(72, 144, size=n)  # age in months ~ 6-12 years
    true_slope = 0.02
    true_intercept = 0.5
    noise = rng.standard_normal(n) * 0.1
    y = true_intercept + true_slope * age + noise

    res = residualize_target(y, pd.DataFrame({"age_months": age}))

    assert abs(res["model_coef"]["age_months"] - true_slope) < 0.005
    assert abs(res["intercept"] - true_intercept) < 0.05
    # R^2 should be high since we planted a strong age effect.
    assert res["age_model_r2"] > 0.85
    # Residual standard deviation should approach the injected noise sd.
    assert abs(res["y_residual"].std() - noise.std()) < 0.05


def test_residualize_target_zero_age_effect_yields_low_r2():
    rng = np.random.default_rng(1)
    n = 100
    age = rng.uniform(72, 144, size=n)
    y = rng.standard_normal(n)  # independent of age

    res = residualize_target(y, pd.DataFrame({"age_months": age}))

    assert res["age_model_r2"] < 0.15
    # Residual ~ y after demeaning since slope ~ 0.
    assert abs(res["y_residual"].mean()) < 0.05


def test_residualize_target_rejects_too_few_samples():
    with pytest.raises(ValueError):
        residualize_target([1.0, 2.0], pd.DataFrame({"age": [10.0, 11.0]}))


def test_residualize_target_propagates_nan_when_covariate_missing():
    age = pd.Series([72.0, np.nan, 144.0, 100.0])
    y   = pd.Series([1.0, 2.0, 3.0, 4.0])
    res = residualize_target(y, pd.DataFrame({"age_months": age}))
    # Residual at the NaN-covariate row must be NaN.
    assert np.isnan(res["y_residual"].iloc[1])
    # Other residuals are finite.
    assert res["y_residual"].iloc[[0, 2, 3]].notna().all()


# ----------------------------------------------------- derive_clinical_threshold

def _prevalence_within(actual, expected, n, tol_samples=1):
    expected_n = expected * n
    return abs(actual * n - expected_n) <= tol_samples


def test_derive_threshold_tertile_bottom_prevalence():
    rng = np.random.default_rng(2)
    n = 60
    y = rng.standard_normal(n)
    out = derive_clinical_threshold(y, method="tertile_bottom")
    assert _prevalence_within(out["prevalence"], 1.0 / 3.0, n)


def test_derive_threshold_quartile_bottom_prevalence():
    rng = np.random.default_rng(3)
    n = 80
    y = rng.standard_normal(n)
    out = derive_clinical_threshold(y, method="quartile_bottom")
    assert _prevalence_within(out["prevalence"], 0.25, n)


def test_derive_threshold_median_prevalence():
    rng = np.random.default_rng(4)
    n = 100
    y = rng.standard_normal(n)
    out = derive_clinical_threshold(y, method="median")
    assert _prevalence_within(out["prevalence"], 0.5, n)


def test_derive_threshold_direction_low_residual_is_atrisk():
    # Hand-build a residual where the lowest two values clearly fall in
    # the bottom tertile and must be flagged 1.
    y = pd.Series([-5.0, -4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = derive_clinical_threshold(y, method="tertile_bottom")
    assert out["y_binary"].iloc[0] == 1
    assert out["y_binary"].iloc[1] == 1
    assert out["y_binary"].iloc[-1] == 0
    assert out["y_binary"].iloc[-2] == 0


def test_derive_threshold_unknown_method_raises():
    with pytest.raises(ValueError):
        derive_clinical_threshold([1.0, 2.0, 3.0], method="bogus")


def test_derive_threshold_propagates_nan():
    y = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    out = derive_clinical_threshold(y, method="median")
    # NaN row must remain NaN in the binary output.
    assert pd.isna(out["y_binary"].iloc[2])
