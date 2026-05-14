"""
Stage 4 — Target Preparation
============================
Two helpers used before regression:

  1. residualize_target   — regress age (or other covariates) out of a
                            continuous target. EF develops sharply between
                            ages 6 and 12, so a "low" raw score in a
                            12-year-old is very different clinically from
                            the same raw score in a 7-year-old. We model
                            the typical age trajectory and analyse the
                            residual ("performance relative to peers").

  2. derive_clinical_threshold — apply a fixed quantile cut to the
                            residualized target to produce a binary
                            at-risk label for downstream screening
                            metrics. Direction convention: 1 = at-risk
                            (lower residual = greater developmental delay
                            in the EF sense).

Both functions return dicts with diagnostics so the QC report can
inspect them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


_THRESHOLD_QUANTILES = {
    "tertile_bottom":  1.0 / 3.0,
    "quartile_bottom": 0.25,
    "median":          0.5,
}


def residualize_target(y, covariates_df):
    """
    Regress ``covariates_df`` out of ``y`` and return the residual.

    Args:
        y: 1-D array-like of the continuous target.
        covariates_df: DataFrame, shape (n_samples, n_covariates). Must be
                       aligned with ``y``. Default in callers: a single
                       ``age_months`` column.

    Returns:
        dict with keys:
            y_residual:    pd.Series of residuals, same index as input.
            age_model_r2:  R^2 of the covariate model on raw y.
            model_coef:    dict mapping each covariate name to its coefficient.
            intercept:     model intercept.
            n_used:        number of rows used (after dropping NaNs).
    """
    if not isinstance(covariates_df, pd.DataFrame):
        covariates_df = pd.DataFrame(covariates_df)

    if isinstance(y, pd.Series):
        y_series = y.copy()
    else:
        y_series = pd.Series(np.asarray(y), index=covariates_df.index)

    X = covariates_df.copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    y_num = pd.to_numeric(y_series, errors="coerce")

    mask = X.notna().all(axis=1) & y_num.notna()
    n_used = int(mask.sum())
    if n_used < 3:
        raise ValueError(
            f"Need at least 3 non-NaN samples to residualize; got {n_used}"
        )

    X_fit = X.loc[mask].values
    y_fit = y_num.loc[mask].values

    model = LinearRegression()
    model.fit(X_fit, y_fit)
    r2 = float(model.score(X_fit, y_fit))

    # Predict for every row that has covariate data (even if y was NaN —
    # downstream code can still subtract). Rows missing covariates get NaN.
    pred = pd.Series(np.nan, index=y_series.index, dtype=float)
    full_mask = X.notna().all(axis=1)
    if full_mask.any():
        pred.loc[full_mask] = model.predict(X.loc[full_mask].values)

    residual = y_num - pred

    coef = {col: float(c) for col, c in zip(covariates_df.columns, model.coef_)}

    return {
        "y_residual": residual,
        "age_model_r2": r2,
        "model_coef": coef,
        "intercept": float(model.intercept_),
        "n_used": n_used,
    }


def derive_clinical_threshold(y_residual, method: str = "tertile_bottom"):
    """
    Cut ``y_residual`` at a fixed quantile to define an at-risk binary label.

    Direction:  ``y_binary == 1`` when ``y_residual <= threshold``
                i.e. lower (more delayed) residual → at-risk.

    Args:
        y_residual: 1-D array-like of residualized target values. NaNs are
                    excluded from the threshold calculation but propagated
                    as NaN in the returned y_binary.
        method: one of {"tertile_bottom", "quartile_bottom", "median"}.

    Returns:
        dict with keys:
            threshold:  float quantile cutpoint.
            y_binary:   pd.Series of {0, 1, NaN}, same index as input.
            prevalence: fraction of non-NaN samples flagged 1.
            method:     echoed method string.
            quantile:   the quantile used (0.333, 0.25, 0.5).
    """
    if method not in _THRESHOLD_QUANTILES:
        raise ValueError(
            f"Unknown threshold method '{method}'. "
            f"Expected one of: {sorted(_THRESHOLD_QUANTILES)}"
        )
    q = _THRESHOLD_QUANTILES[method]

    if isinstance(y_residual, pd.Series):
        series = y_residual.copy()
    else:
        series = pd.Series(np.asarray(y_residual, dtype=float))

    valid = series.dropna()
    if valid.empty:
        raise ValueError("y_residual is all-NaN; cannot derive threshold")
    threshold = float(np.quantile(valid.values, q))

    y_binary = pd.Series(np.nan, index=series.index, dtype=float)
    notna = series.notna()
    y_binary.loc[notna] = (series.loc[notna] <= threshold).astype(int)

    n_total = int(notna.sum())
    n_pos = int((y_binary == 1).sum())
    prevalence = n_pos / n_total if n_total else float("nan")

    return {
        "threshold": threshold,
        "y_binary":  y_binary,
        "prevalence": prevalence,
        "method": method,
        "quantile": q,
    }
