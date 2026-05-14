"""
Stage 4 — Unsupervised Feature Curation
=======================================
Two passes that shrink a wide feature matrix without ever looking at the
target, so they can run once on the full cohort before CV without leaking
labels into the training fold.

  1. drop_low_variance        — remove near-constant columns (variance < eps).
  2. drop_collinear_hierarchical — cluster features whose pairwise |corr|
                                exceeds a threshold and keep one
                                representative (highest individual variance)
                                per cluster.

Both functions return a dict with the filtered matrix plus diagnostics
(kept/dropped indices, cluster_map) for the QC report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def drop_low_variance(X, threshold: float = 1e-6):
    """
    Drop columns with variance below ``threshold``.

    Args:
        X: 2-D ndarray or DataFrame, shape (n_samples, n_features).
        threshold: variance floor. Columns with var <= threshold are removed.

    Returns:
        dict with keys:
            X_filtered:  filtered matrix in the same container type as ``X``.
            kept_idx:    np.ndarray of int column indices retained.
            dropped_idx: np.ndarray of int column indices removed.
            variances:   np.ndarray of per-column variances.
    """
    is_df = isinstance(X, pd.DataFrame)
    arr = X.values if is_df else np.asarray(X)
    arr = arr.astype(float, copy=False)

    # nan-safe variance: treat NaN as missing rather than poisoning the column.
    variances = np.nanvar(arr, axis=0, ddof=0)
    # Columns that are entirely NaN have variance = NaN; treat them as zero-var.
    variances = np.where(np.isnan(variances), 0.0, variances)

    keep_mask = variances > threshold
    kept_idx = np.flatnonzero(keep_mask)
    dropped_idx = np.flatnonzero(~keep_mask)

    if is_df:
        X_filtered = X.iloc[:, kept_idx].copy()
    else:
        X_filtered = arr[:, kept_idx]

    return {
        "X_filtered": X_filtered,
        "kept_idx": kept_idx,
        "dropped_idx": dropped_idx,
        "variances": variances,
    }


def drop_collinear_hierarchical(X, feature_names, corr_threshold: float = 0.95):
    """
    Drop redundant columns by hierarchical clustering on (1 - |corr|).

    Distance metric:  d(i, j) = 1 - |corr(i, j)|
    Linkage:          average
    Cluster cut:      criterion='distance', t = 1 - corr_threshold
                      → features with pairwise |corr| >= corr_threshold
                        end up in the same cluster.

    Per cluster we keep the column with the highest individual variance
    (tie-broken by leftmost original index for determinism). The rest are
    dropped. Cluster_map records each kept feature's followers so the QC
    report can show what was merged into what.

    Args:
        X: 2-D ndarray or DataFrame, shape (n_samples, n_features).
        feature_names: list/Index of column names, length == n_features.
        corr_threshold: |corr| >= this value collapses features together.

    Returns:
        dict with keys:
            X_filtered:  matrix containing only kept columns (same type as X).
            kept_names:  list of kept feature names, in original column order.
            dropped_names: list of dropped feature names, in original order.
            cluster_map: dict {kept_name: [dropped_names_in_same_cluster, ...]}.
    """
    is_df = isinstance(X, pd.DataFrame)
    arr = X.values if is_df else np.asarray(X)
    arr = arr.astype(float, copy=False)

    feature_names = list(feature_names)
    n_features = arr.shape[1]
    if n_features != len(feature_names):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) does not match "
            f"X.shape[1] ({n_features})"
        )

    # Edge cases: 0 or 1 feature → nothing to drop.
    if n_features <= 1:
        kept = list(feature_names)
        if is_df:
            X_filtered = X.copy()
        else:
            X_filtered = arr.copy()
        return {
            "X_filtered": X_filtered,
            "kept_names": kept,
            "dropped_names": [],
            "cluster_map": {name: [] for name in kept},
        }

    variances = np.nanvar(arr, axis=0, ddof=0)
    variances = np.where(np.isnan(variances), 0.0, variances)

    # Pairwise correlation. NaNs are masked column-wise via pd.DataFrame.corr.
    corr = pd.DataFrame(arr, columns=feature_names).corr().values
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    distance = 1.0 - np.abs(corr)
    # Numerical noise can leave tiny negatives or asymmetry — clean for squareform.
    distance = np.clip(distance, 0.0, 2.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)

    # squareform requires a strictly hollow symmetric matrix.
    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=1.0 - corr_threshold, criterion="distance")

    cluster_map: dict[str, list[str]] = {}
    kept_idx: list[int] = []

    for cid in np.unique(cluster_ids):
        members = np.flatnonzero(cluster_ids == cid)
        if len(members) == 1:
            keep = int(members[0])
        else:
            # Highest-variance representative; ties broken by smallest index.
            order = sorted(members.tolist(), key=lambda i: (-variances[i], i))
            keep = order[0]
        kept_idx.append(keep)
        followers = [feature_names[m] for m in members if m != keep]
        cluster_map[feature_names[keep]] = followers

    kept_idx = sorted(kept_idx)
    kept_names = [feature_names[i] for i in kept_idx]
    dropped_names = [name for name in feature_names if name not in set(kept_names)]

    if is_df:
        X_filtered = X.iloc[:, kept_idx].copy()
    else:
        X_filtered = arr[:, kept_idx]

    return {
        "X_filtered": X_filtered,
        "kept_names": kept_names,
        "dropped_names": dropped_names,
        "cluster_map": cluster_map,
    }
