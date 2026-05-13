"""
Stage 6: Explicit Density-Matrix Features
==========================================

Path B of the Density-Matrix Brainwaves project. Stage 5 (exploratory_quantum.py)
runs a quantum kernel SVM on quantum-inspired summary features (QEPP, von Neumann
entropy, etc.); Stage 6 instead builds an explicit per-band density matrix from
the multichannel analytic signal and uses the matrix entries directly as
features. The two should converge under the Schuld (2021) kernel-equivalence:

    K_Q(x, y) = |<psi(x)|psi(y)>|^2 = Tr(rho_x rho_y).

Construction (per subject, per band b in {delta, theta, alpha, beta}):

  1. Zero-phase bandpass each of the N=15 channels (sosfiltfilt).
  2. Compute the analytic signal channel-wise via the Hilbert transform.
     At time t this yields a complex vector psi(t) in C^N.
  3. Normalize each time-vector to unit norm (state normalization):
         psi_hat(t) = psi(t) / ||psi(t)||_2
  4. Average outer products over all time samples across all epochs:
         rho_b = (1/T_total) sum_t |psi_hat(t)><psi_hat(t)|
  5. Symmetrize (rho_b = (rho_b + rho_b^H)/2) and renormalize so Tr(rho_b)=1.

The result is a Hermitian, positive semi-definite, trace-one matrix per band.

Feature vector per subject:
  - For each band: N real diagonal entries (occupation probabilities)
                 + (Re, Im) parts of the strictly upper triangle (N*(N-1) reals)
  - Real (Re, Im) flattening is mathematically equivalent to (mag, phase) but
    avoids the cyclic-angle pathology that breaks linear/SVM classifiers.
  - Total features: 4 bands * N^2 = 4 * 225 = 900 reals at N=15.

Two evaluation tracks:
  (a) Generic classifiers (LogReg / RF / SVM-RBF / MLP) on the flattened vector
      under matched 10-fold x 10-repeat stratified CV.
  (b) An SVM with a custom precomputed Hilbert-Schmidt kernel
         K(s, t) = (1/B) sum_b Tr(rho_b^s rho_b^t)
      built directly from the density matrices. This is the *exact* analogue
      of the QSVM fidelity kernel and is the cleanest test of Schuld's
      equivalence: vectorizing rho and taking the standard inner product is
      identical to the Hilbert-Schmidt inner product, so a linear SVM on the
      flat features should match (b) to numerical precision.

Outputs (in config.paths.output_dir):
  - density_matrix_features.csv     — one row per subject, ~900 columns
  - density_matrices.npz            — raw complex 15x15 matrices (subject x band)
  - density_matrix_results.csv      — model x feature_set x CV scores
  - density_matrix_vs_qsvm.csv      — paired per-fold deltas vs QSVM
  - density_matrix_qc.json          — sanity-check report

References:
  Schuld, M. (2021). Supervised quantum machine learning models are kernel
    methods. arXiv:2101.11020.
  Havlicek, V. et al. (2019). Supervised learning with quantum-enhanced
    feature spaces. Nature 567, 209-212.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sig

warnings.filterwarnings("ignore")
logger = logging.getLogger("biomarker_iium.stage6")


# ──────────────────────────────────────────────────────────────────
# Density-matrix construction
# ──────────────────────────────────────────────────────────────────

def _bandpass_zero_phase(x: np.ndarray, fmin: float, fmax: float, sfreq: float,
                         order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass along the last axis.

    sosfiltfilt is mandatory for analytic-signal phase features: a causal
    sosfilt would inject a frequency-dependent group delay that contaminates
    every off-diagonal phase term in the density matrix.
    """
    sos = sig.butter(order, [fmin, fmax], btype="band", fs=sfreq, output="sos")
    return sig.sosfiltfilt(sos, x, axis=-1)


def compute_density_matrix(epochs, fmin: float, fmax: float) -> np.ndarray:
    """Build the band-limited density matrix rho_b for one subject/condition.

    Parameters
    ----------
    epochs : mne.Epochs
        Cleaned epochs, shape (n_epochs, n_channels, n_times).
    fmin, fmax : float
        Bandpass edges in Hz.

    Returns
    -------
    rho : ndarray of shape (n_channels, n_channels), complex128
        Hermitian, positive semi-definite, trace-one density matrix.
    """
    data = epochs.get_data().astype(np.float64)        # (E, C, T)
    sfreq = float(epochs.info["sfreq"])
    n_epochs, n_channels, n_times = data.shape

    # Zero-phase bandpass per epoch per channel.
    filtered = _bandpass_zero_phase(data, fmin, fmax, sfreq)

    # Analytic signal along time. scipy.signal.hilbert acts on the last axis.
    analytic = sig.hilbert(filtered, axis=-1)          # complex (E, C, T)

    # Stack epochs along time so we get one long C x T_total array.
    psi = analytic.transpose(1, 0, 2).reshape(n_channels, -1)   # (C, T_total)

    # Per-time-point unit-norm state vector. Drop columns whose norm is
    # numerically zero (silent channels at this frequency band would otherwise
    # divide by 1e-16 noise and dominate the average).
    norms = np.linalg.norm(psi, axis=0)                # (T_total,)
    keep = norms > 1e-12
    if keep.sum() == 0:
        return np.zeros((n_channels, n_channels), dtype=np.complex128)

    psi = psi[:, keep] / norms[keep]                   # (C, T_kept)

    # Average outer product: rho = (1/T) sum_t psi_t psi_t^H.
    # Vectorized as (C, T) @ (T, C) with conjugate.
    rho = (psi @ psi.conj().T) / psi.shape[1]

    # Force Hermitian symmetry (numerical noise) and unit trace.
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.real(np.trace(rho))
    if tr > 0:
        rho = rho / tr
    return rho


def density_matrix_qc(rho: np.ndarray, atol: float = 1e-8) -> dict:
    """Sanity-check a density matrix. Returns a small QC dict."""
    n = rho.shape[0]
    herm_err = float(np.max(np.abs(rho - rho.conj().T)))
    trace_err = float(abs(np.real(np.trace(rho)) - 1.0))
    eigvals = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    min_eig = float(eigvals.min())
    # Cauchy-Schwarz: |rho_ij|^2 <= rho_ii * rho_jj
    diag = np.real(np.diag(rho))
    cs_violation = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            bound = diag[i] * diag[j]
            if np.abs(rho[i, j]) ** 2 > bound + atol:
                cs_violation = max(cs_violation,
                                   float(np.abs(rho[i, j]) ** 2 - bound))
    return {
        "hermiticity_error": herm_err,
        "trace_error": trace_err,
        "min_eigenvalue": min_eig,
        "cs_violation": cs_violation,
    }


def flatten_density_matrix(rho: np.ndarray, ch_names: List[str],
                           band: str) -> Dict[str, float]:
    """Flatten one density matrix into a labeled dict of real features.

    - n diagonals (real): rho_ii
    - n(n-1)/2 off-diagonals split into Re and Im parts.
    """
    n = rho.shape[0]
    out: Dict[str, float] = {}
    diag = np.real(np.diag(rho))
    for i in range(n):
        out[f"dm_{band}_diag_{ch_names[i]}"] = float(diag[i])
    for i in range(n):
        for j in range(i + 1, n):
            out[f"dm_{band}_re_{ch_names[i]}_{ch_names[j]}"] = float(np.real(rho[i, j]))
            out[f"dm_{band}_im_{ch_names[i]}_{ch_names[j]}"] = float(np.imag(rho[i, j]))
    return out


# ──────────────────────────────────────────────────────────────────
# Subject-level extraction
# ──────────────────────────────────────────────────────────────────

def extract_density_features(epochs_dict, config) -> Tuple[pd.DataFrame, dict, dict]:
    """Build density matrices for every subject and every band.

    Returns
    -------
    features_df : DataFrame (one row per subject, ~900 columns)
    matrices    : dict subject_id -> {band: complex matrix}
    qc          : dict subject_id -> {band: qc dict}
    """
    bands = config["features"]["bands"]

    # Group by subject.
    subjects: Dict[str, dict] = {}
    for (sub, cond), ep in epochs_dict.items():
        subjects.setdefault(sub, {})[cond] = ep

    rows: List[Dict[str, float]] = []
    matrices: Dict[str, Dict[str, np.ndarray]] = {}
    qc: Dict[str, Dict[str, dict]] = {}

    for sub_id in sorted(subjects.keys()):
        cond_map = subjects[sub_id]
        primary = "Eyes_Open" if "Eyes_Open" in cond_map else next(iter(cond_map))
        epochs = cond_map[primary]
        ch_names = list(epochs.ch_names)
        row = {"subject_id": sub_id, "condition": primary}

        sub_matrices: Dict[str, np.ndarray] = {}
        sub_qc: Dict[str, dict] = {}

        for band, (fmin, fmax) in bands.items():
            rho = compute_density_matrix(epochs, fmin, fmax)
            sub_matrices[band] = rho
            sub_qc[band] = density_matrix_qc(rho)
            row.update(flatten_density_matrix(rho, ch_names, band))

        matrices[sub_id] = sub_matrices
        qc[sub_id] = sub_qc
        rows.append(row)
        print(f"  density-matrix: {sub_id}  bands={list(sub_matrices)}  "
              f"trace_err_max={max(q['trace_error'] for q in sub_qc.values()):.2e}  "
              f"min_eig={min(q['min_eigenvalue'] for q in sub_qc.values()):.2e}")

    return pd.DataFrame(rows), matrices, qc


# ──────────────────────────────────────────────────────────────────
# Hilbert–Schmidt kernel
# ──────────────────────────────────────────────────────────────────

def hilbert_schmidt_kernel(matrices_x: List[Dict[str, np.ndarray]],
                           matrices_y: List[Dict[str, np.ndarray]],
                           bands: List[str]) -> np.ndarray:
    """K[i, j] = (1/B) sum_b Re Tr(rho_b^i rho_b^j).

    For Hermitian rho, Tr(A B) is real. We take Re explicitly to absorb
    floating-point imaginary noise.
    """
    n_x, n_y = len(matrices_x), len(matrices_y)
    K = np.zeros((n_x, n_y), dtype=np.float64)
    for b in bands:
        # Pre-vectorize once per band: vec(rho) so K_b = V_x @ V_y^H gives
        # exactly the Hilbert-Schmidt kernel. Doing it this way avoids an
        # n_x * n_y * O(N^3) loop.
        V_x = np.stack([m[b].reshape(-1) for m in matrices_x])  # (n_x, N^2)
        V_y = np.stack([m[b].reshape(-1) for m in matrices_y])
        K_b = np.real(V_x @ V_y.conj().T)
        K += K_b
    K /= len(bands)
    return K


class HilbertSchmidtKernelSVM:
    """SVM with K(s, t) = (1/B) sum_b Tr(rho_b^s rho_b^t).

    Stores the training-set list of band->matrix dicts and rebuilds the kernel
    to test points at predict time. Tiny by design (N=28); no caching needed.
    """

    def __init__(self, bands: List[str], C: float = 1.0):
        self.bands = list(bands)
        self.C = C

    def fit(self, matrices: List[Dict[str, np.ndarray]], y) -> "HilbertSchmidtKernelSVM":
        from sklearn.svm import SVC
        self._train = matrices
        K = hilbert_schmidt_kernel(matrices, matrices, self.bands)
        K = 0.5 * (K + K.T)
        self._svm = SVC(kernel="precomputed", C=self.C, probability=True,
                         random_state=42)
        self._svm.fit(K, y)
        return self

    def _kernel_to_train(self, matrices: List[Dict[str, np.ndarray]]) -> np.ndarray:
        return hilbert_schmidt_kernel(matrices, self._train, self.bands)

    def predict(self, matrices: List[Dict[str, np.ndarray]]) -> np.ndarray:
        return self._svm.predict(self._kernel_to_train(matrices))

    def predict_proba(self, matrices: List[Dict[str, np.ndarray]]) -> np.ndarray:
        return self._svm.predict_proba(self._kernel_to_train(matrices))


# ──────────────────────────────────────────────────────────────────
# Matched cross-validation
# ──────────────────────────────────────────────────────────────────

def _build_classifier_zoo(random_state: int = 42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    return [
        ("LogReg",  LogisticRegression(max_iter=2000, C=0.1,
                                        random_state=random_state)),
        ("SVM_RBF", SVC(kernel="rbf", C=1.0, probability=True,
                         random_state=random_state)),
        ("SVM_Lin", SVC(kernel="linear", C=1.0, probability=True,
                         random_state=random_state)),
        ("RF",      RandomForestClassifier(n_estimators=200, max_depth=3,
                                           random_state=random_state)),
        ("MLP",     MLPClassifier(hidden_layer_sizes=(64,),
                                   max_iter=500, early_stopping=True,
                                   random_state=random_state)),
    ]


def _evaluate_flat_features(X: np.ndarray, y: np.ndarray, splits, models,
                            tag: str, n_select: int) -> List[dict]:
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rows: List[dict] = []
    for name, model in models:
        per_fold_acc, per_fold_auc = [], []
        for tr_idx, te_idx in splits:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=min(n_select, X.shape[1]))),
                ("clf", model),
            ])
            pipe.fit(X[tr_idx], y[tr_idx])
            yhat = pipe.predict(X[te_idx])
            try:
                yprob = pipe.predict_proba(X[te_idx])[:, 1]
                auc = roc_auc_score(y[te_idx], yprob)
            except Exception:
                auc = np.nan
            per_fold_acc.append(balanced_accuracy_score(y[te_idx], yhat))
            per_fold_auc.append(auc)
        rows.append({
            "feature_set": tag,
            "model": name,
            "balanced_accuracy_mean": float(np.mean(per_fold_acc)),
            "balanced_accuracy_std": float(np.std(per_fold_acc)),
            "roc_auc_mean": float(np.nanmean(per_fold_auc)),
            "n_features": int(X.shape[1]),
        })
    return rows


def _evaluate_hs_kernel(matrices: List[Dict[str, np.ndarray]], y, splits,
                        bands: List[str], C: float = 1.0) -> List[dict]:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    per_fold_acc, per_fold_auc = [], []
    for tr_idx, te_idx in splits:
        clf = HilbertSchmidtKernelSVM(bands=bands, C=C)
        clf.fit([matrices[i] for i in tr_idx], y[tr_idx])
        yhat = clf.predict([matrices[i] for i in te_idx])
        try:
            yprob = clf.predict_proba([matrices[i] for i in te_idx])[:, 1]
            auc = roc_auc_score(y[te_idx], yprob)
        except Exception:
            auc = np.nan
        per_fold_acc.append(balanced_accuracy_score(y[te_idx], yhat))
        per_fold_auc.append(auc)
    return [{
        "feature_set": "density_matrix",
        "model": "HS_kernel_SVM",
        "balanced_accuracy_mean": float(np.mean(per_fold_acc)),
        "balanced_accuracy_std": float(np.std(per_fold_acc)),
        "roc_auc_mean": float(np.nanmean(per_fold_auc)),
        "n_features": int(sum(m[bands[0]].size for m in matrices[:1]) * len(bands)),
    }], per_fold_acc


def _evaluate_qsvm_matched(X: np.ndarray, y, splits, n_qubits: int = 6,
                            n_layers: int = 2, C: float = 1.0):
    """Re-run QSVM under the *same* CV splits used for density-matrix tests."""
    try:
        from stages.qsvm_classifier import QuantumKernelSVM
    except ImportError:
        return None, None

    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    per_fold_acc, per_fold_auc = [], []
    for tr_idx, te_idx in splits:
        clf = QuantumKernelSVM(n_qubits=n_qubits, n_layers=n_layers,
                                C=C, use_entangling=True)
        clf.fit(X[tr_idx], y[tr_idx])
        yhat = clf.predict(X[te_idx])
        try:
            yprob = clf.predict_proba(X[te_idx])[:, 1]
            auc = roc_auc_score(y[te_idx], yprob)
        except Exception:
            auc = np.nan
        per_fold_acc.append(balanced_accuracy_score(y[te_idx], yhat))
        per_fold_auc.append(auc)
    return [{
        "feature_set": "quantum_features",
        "model": f"QSVM_{n_qubits}q_ZZ",
        "balanced_accuracy_mean": float(np.mean(per_fold_acc)),
        "balanced_accuracy_std": float(np.std(per_fold_acc)),
        "roc_auc_mean": float(np.nanmean(per_fold_auc)),
        "n_features": int(X.shape[1]),
    }], per_fold_acc


# ──────────────────────────────────────────────────────────────────
# Stage orchestrator
# ──────────────────────────────────────────────────────────────────

def run_stage6(config, epochs_dict, full_df: pd.DataFrame,
               quantum_df: pd.DataFrame | None = None,
               n_splits: int = 10, n_repeats: int = 10,
               random_state: int = 42) -> pd.DataFrame:
    """End-to-end Stage 6 driver.

    Parameters
    ----------
    config : dict
    epochs_dict : {(subject_id, condition): mne.Epochs}
    full_df : DataFrame produced by Stage 3 (subject_id, ef_group_global, ...)
    quantum_df : optional DataFrame produced by Stage 5. If provided, the
        QSVM is re-evaluated under the same matched CV splits. If None, this
        comparison is skipped.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    print("\n" + "=" * 60)
    print("STAGE 6: Explicit Density-Matrix Features (Path B)")
    print("=" * 60)

    bands = list(config["features"]["bands"].keys())

    # 1. Build density matrices.
    dm_df, matrices, qc = extract_density_features(epochs_dict, config)

    # 2. Merge labels.
    target = "ef_group_global"
    labels = full_df[["subject_id", target]].dropna()
    merged = dm_df.merge(labels, on="subject_id", how="inner")
    if len(merged) < n_splits:
        raise ValueError(
            f"Stage 6 needs at least {n_splits} labelled subjects; got {len(merged)}"
        )

    feat_cols = [c for c in merged.columns if c.startswith("dm_")]
    X_dm = merged[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y = merged[target].astype(int).to_numpy()
    subject_order = merged["subject_id"].tolist()
    matrices_ordered = [matrices[s] for s in subject_order]

    # 3. Generate matched CV splits ONCE; reuse for every model.
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                  random_state=random_state)
    splits = list(cv.split(np.zeros(len(y)), y))
    print(f"  matched CV: {n_splits}-fold x {n_repeats} repeats "
          f"({len(splits)} folds)  N={len(y)}  feat={X_dm.shape[1]}")

    # 4. Evaluate the flat-feature classifiers.
    models = _build_classifier_zoo(random_state=random_state)
    rows = _evaluate_flat_features(
        X_dm, y, splits, models, tag="density_matrix", n_select=10
    )

    # 5. Evaluate the Hilbert-Schmidt kernel SVM.
    hs_rows, hs_per_fold = _evaluate_hs_kernel(matrices_ordered, y, splits, bands)
    rows.extend(hs_rows)

    # 6. Optional matched QSVM comparison on Stage 5 quantum features.
    qsvm_per_fold = None
    if quantum_df is not None:
        q_merged = quantum_df.merge(labels, on="subject_id", how="inner")
        q_merged = q_merged.set_index("subject_id").loc[subject_order]
        q_cols = [c for c in q_merged.columns if c != target]
        X_q = q_merged[q_cols].fillna(0.0).to_numpy(dtype=np.float64)
        qsvm_rows, qsvm_per_fold = _evaluate_qsvm_matched(X_q, y, splits)
        if qsvm_rows is not None:
            rows.extend(qsvm_rows)

    # 7. Classical baseline: PSD/coh/cov features from Stage 4 dataset, same CV.
    classical_cols = [c for c in full_df.columns
                      if c.startswith(("psd_", "coh_", "cov_", "tbr_", "faa_",
                                       "alpha_reactivity"))]
    if classical_cols:
        cl_df = full_df.set_index("subject_id").loc[subject_order, classical_cols]
        X_cl = cl_df.fillna(0.0).to_numpy(dtype=np.float64)
        rows.extend(_evaluate_flat_features(
            X_cl, y, splits, models, tag="classical", n_select=10
        ))

    results_df = pd.DataFrame(rows).sort_values(
        ["feature_set", "balanced_accuracy_mean"], ascending=[True, False]
    )

    # 8. Persist outputs.
    out_dir = config["paths"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    dm_df.to_csv(os.path.join(out_dir, "density_matrix_features.csv"), index=False)
    results_df.to_csv(os.path.join(out_dir, "density_matrix_results.csv"),
                       index=False)

    # Raw matrices for downstream replication.
    np.savez(
        os.path.join(out_dir, "density_matrices.npz"),
        **{
            f"{sub}_{band}": matrices[sub][band]
            for sub in matrices for band in matrices[sub]
        },
    )

    with open(os.path.join(out_dir, "density_matrix_qc.json"), "w") as f:
        # Convert numpy floats to plain floats for JSON serialization.
        qc_clean = {sub: {b: {k: float(v) for k, v in qb.items()}
                          for b, qb in bandqc.items()}
                    for sub, bandqc in qc.items()}
        json.dump(qc_clean, f, indent=2)

    # 9. Paired comparison vs QSVM (if we have per-fold scores from both).
    if qsvm_per_fold is not None:
        deltas = np.array(hs_per_fold) - np.array(qsvm_per_fold)
        from scipy.stats import wilcoxon
        try:
            stat, pval = wilcoxon(hs_per_fold, qsvm_per_fold,
                                   zero_method="zsplit")
        except ValueError:
            stat, pval = float("nan"), float("nan")
        pair = pd.DataFrame({
            "fold": np.arange(len(deltas)),
            "hs_kernel_acc": hs_per_fold,
            "qsvm_acc": qsvm_per_fold,
            "delta_hs_minus_qsvm": deltas,
        })
        pair.to_csv(os.path.join(out_dir, "density_matrix_vs_qsvm.csv"),
                     index=False)
        mean_delta = float(np.mean(deltas))
        print(f"\n  paired delta (HS_kernel_SVM - QSVM): "
              f"mean={mean_delta:+.4f}  Wilcoxon p={pval:.3f}")
        if abs(mean_delta) < 0.02:
            print("  Schuld 2021 prediction supported: the two kernels match "
                  "within +/- 2 pp.")
        else:
            print("  Mean delta exceeds +/- 2 pp -- the encoding circuit and "
                  "the explicit density matrix do NOT span equivalent kernels "
                  "on this data; investigate the Stage 5 PCA + [0, pi] "
                  "rescaling step.")

    # 10. Print summary table.
    print("\n  --- Stage 6 results (matched 10-fold x 10) ---")
    for _, r in results_df.iterrows():
        print(f"  {r['feature_set']:18s} | {r['model']:14s} | "
              f"bacc={r['balanced_accuracy_mean']:.3f} "
              f"(+/-{r['balanced_accuracy_std']:.3f})  "
              f"auc={r['roc_auc_mean']:.3f}")

    print(f"\n  saved: {out_dir}/density_matrix_features.csv")
    print(f"  saved: {out_dir}/density_matrix_results.csv")
    print(f"  saved: {out_dir}/density_matrices.npz")
    print(f"  saved: {out_dir}/density_matrix_qc.json")
    if qsvm_per_fold is not None:
        print(f"  saved: {out_dir}/density_matrix_vs_qsvm.csv")

    return results_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.io import load_config
    import mne

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = load_config()

    epoch_dir = os.path.join(config["paths"]["output_dir"], "cleaned_epochs")
    all_epochs = {}
    for f in sorted(os.listdir(epoch_dir)):
        if not f.endswith("-epo.fif"):
            continue
        parts = f.replace("-epo.fif", "").split("_", 1)
        all_epochs[(parts[0], parts[1])] = mne.read_epochs(
            os.path.join(epoch_dir, f), verbose=False
        )

    full = pd.read_csv(os.path.join(config["paths"]["output_dir"],
                                     "full_dataset.csv"))

    quantum_path = os.path.join(config["paths"]["output_dir"],
                                 "quantum_features.csv")
    quantum = pd.read_csv(quantum_path) if os.path.exists(quantum_path) else None

    run_stage6(config, all_epochs, full, quantum_df=quantum)
