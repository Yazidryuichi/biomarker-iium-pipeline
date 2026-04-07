"""
Exploratory: Quantum-Inspired Feature Engineering for EEG
==========================================================

Applies quantum-inspired methods to QEEG data as a bridge to
the PhD proposal (Studies 2-3). Three approaches:

1. QEPP (Quantum Entangled Particles Pattern)
   - Treats EEG channel pairs as coupled quantum systems
   - Computes interference patterns between channels
   - Based on: Alotaibi et al. (2026), Scientific Reports

2. Quantum Probability Feature Interactions
   - Models EEG feature combinations using quantum probability
   - Tests for interference terms (non-classical interactions)
   - Based on: Busemeyer & Bruza (2012)

3. Tensor Network Decomposition
   - Represents multi-channel EEG as a tensor network
   - Extracts entanglement-like features between channel groups
   - Captures inter-channel dependencies standard methods miss

These are EXPLORATORY. Results should be compared against
classical baselines from Stage 4.

References:
  Alotaibi et al. (2026). Quantum inspired feature engineering
    for EEG signal classification. Scientific Reports, 16.
  Busemeyer & Bruza (2012). Quantum Models of Cognition and
    Decision. Cambridge University Press.
  Khrennikov & Yamada (2025). Quantum-like representation of
    neuronal networks' activity. Front. Hum. Neurosci., 19.
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import signal as sig
from scipy.linalg import expm, logm

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────
# 1. QEPP: Quantum Entangled Particles Pattern
# ──────────────────────────────────────────────────────────────────

def compute_qepp_features(epochs, bands):
    """
    Quantum Entangled Particles Pattern (QEPP).

    Treats each EEG channel pair as a coupled quantum system.
    For each pair (i, j) in a frequency band:
      1. Compute the analytic signal (Hilbert transform)
      2. Represent each channel as a quantum state vector
      3. Compute the tensor product (joint state)
      4. Measure interference: deviation from product state

    The interference term captures inter-channel dependencies
    that standard coherence or correlation miss — specifically,
    non-linear phase-amplitude relationships that violate
    classical independence assumptions.

    Returns features per channel pair per band.
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    n_channels = len(ch_names)

    features = {}

    for band_name, (fmin, fmax) in bands.items():
        # Bandpass filter
        sos = sig.butter(4, [fmin, fmax], btype="band", fs=sfreq, output="sos")

        # Compute per-epoch interference, then average
        pair_interference = {}

        for epoch in data:
            # Filter each channel
            filtered = np.array([sig.sosfilt(sos, epoch[ch]) for ch in range(n_channels)])

            # Analytic signal (complex representation = "quantum state")
            analytic = np.array([sig.hilbert(filtered[ch]) for ch in range(n_channels)])

            # Normalize to unit vectors (quantum state normalization)
            norms = np.abs(analytic)
            norms[norms == 0] = 1e-10
            states = analytic / norms

            # Compute interference for selected channel pairs
            # Focus on frontal-central, frontal-parietal pairs
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    pair_key = f"{ch_names[i]}_{ch_names[j]}"

                    # Joint probability: |state_i + state_j|^2
                    joint = np.abs(states[i] + states[j]) ** 2

                    # Classical sum: |state_i|^2 + |state_j|^2
                    classical = np.abs(states[i]) ** 2 + np.abs(states[j]) ** 2

                    # Interference term: joint - classical
                    # This equals 2 * Re(state_i * conj(state_j))
                    interference = joint - classical

                    # Summary statistics of interference
                    mean_interf = np.mean(interference)
                    std_interf = np.std(interference)
                    max_interf = np.max(np.abs(interference))

                    if pair_key not in pair_interference:
                        pair_interference[pair_key] = {
                            "mean": [], "std": [], "max": []
                        }
                    pair_interference[pair_key]["mean"].append(mean_interf)
                    pair_interference[pair_key]["std"].append(std_interf)
                    pair_interference[pair_key]["max"].append(max_interf)

        # Average across epochs — keep only key pairs to limit dimensionality
        key_pairs = [
            "Fz_Cz", "Fz_Pz", "F3_P3", "F4_P4",  # fronto-parietal
            "F3_F4", "C3_C4", "P3_P4",  # inter-hemispheric
            "Fp1_Fz", "Fp2_Fz",  # prefrontal-frontal
            "Fz_O1", "Fz_O2",  # fronto-occipital
        ]
        for pair_key, vals in pair_interference.items():
            if pair_key in key_pairs:
                features[f"qepp_mean_{band_name}_{pair_key}"] = np.mean(vals["mean"])
                features[f"qepp_std_{band_name}_{pair_key}"] = np.mean(vals["std"])
                features[f"qepp_max_{band_name}_{pair_key}"] = np.mean(vals["max"])

    return features


# ──────────────────────────────────────────────────────────────────
# 2. Quantum Probability Feature Interactions
# ──────────────────────────────────────────────────────────────────

def compute_quantum_probability_features(epochs, bands):
    """
    Quantum probability interpretation of EEG feature interactions.

    In classical probability: P(A and B) = P(A) * P(B) for independent events.
    In quantum probability: P(A and B) = P(A) * P(B) + interference_term.

    For EEG: if theta power and beta power at a channel are "incompatible
    observables" (measuring one affects the other), their joint distribution
    should show interference — deviations from the product of marginals.

    This computes:
      1. Marginal distributions of band powers
      2. Joint distributions of band-power pairs
      3. Interference terms = joint - product_of_marginals
      4. Interference magnitude as a feature

    Connection to PhD proposal Study 2: if these interference terms
    predict executive function above classical features alone, it
    suggests quantum probability captures something real about how
    neural oscillations interact.
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    n_channels = len(ch_names)

    features = {}

    # Compute band powers per epoch per channel
    band_powers = {}
    for band_name, (fmin, fmax) in bands.items():
        sos = sig.butter(4, [fmin, fmax], btype="band", fs=sfreq, output="sos")
        powers = []
        for epoch in data:
            filtered = np.array([sig.sosfilt(sos, epoch[ch]) for ch in range(n_channels)])
            power = np.mean(filtered ** 2, axis=1)  # per channel
            powers.append(power)
        band_powers[band_name] = np.array(powers)  # (n_epochs, n_channels)

    # Test for interference between band pairs
    band_pairs = [
        ("theta", "beta"),   # TBR decomposed — the classic EF biomarker
        ("theta", "alpha"),  # theta-alpha interaction
        ("alpha", "beta"),   # alpha-beta transition
        ("delta", "beta"),   # slow-fast coupling
    ]

    n_bins = 10

    for band_a, band_b in band_pairs:
        if band_a not in band_powers or band_b not in band_powers:
            continue

        for ch_idx, ch in enumerate(ch_names):
            # Only frontal and central channels
            if ch not in ["Fz", "F3", "F4", "Cz", "C3", "C4"]:
                continue

            a = band_powers[band_a][:, ch_idx]
            b = band_powers[band_b][:, ch_idx]

            if len(a) < 10:
                continue

            # Discretize into bins
            a_bins = np.digitize(a, np.linspace(a.min(), a.max(), n_bins + 1)[1:-1])
            b_bins = np.digitize(b, np.linspace(b.min(), b.max(), n_bins + 1)[1:-1])

            # Marginal distributions
            p_a = np.bincount(a_bins, minlength=n_bins) / len(a_bins)
            p_b = np.bincount(b_bins, minlength=n_bins) / len(b_bins)

            # Joint distribution
            joint_counts = np.zeros((n_bins, n_bins))
            for ai, bi in zip(a_bins, b_bins):
                if ai < n_bins and bi < n_bins:
                    joint_counts[ai, bi] += 1
            p_joint = joint_counts / joint_counts.sum() if joint_counts.sum() > 0 else joint_counts

            # Classical (independent) prediction
            p_classical = np.outer(p_a, p_b)

            # Interference term
            interference = p_joint - p_classical

            # Features from interference matrix
            features[f"qi_mean_{band_a}_{band_b}_{ch}"] = np.mean(np.abs(interference))
            features[f"qi_max_{band_a}_{band_b}_{ch}"] = np.max(np.abs(interference))
            features[f"qi_frob_{band_a}_{band_b}_{ch}"] = np.linalg.norm(interference, "fro")

            # Total interference magnitude (sum of absolute deviations)
            features[f"qi_total_{band_a}_{band_b}_{ch}"] = np.sum(np.abs(interference))

    return features


# ──────────────────────────────────────────────────────────────────
# 3. Tensor Network / Non-Commutative Features
# ──────────────────────────────────────────────────────────────────

def compute_tensor_features(epochs, bands):
    """
    Tensor-based features treating multi-channel EEG as a quantum-like system.

    Approach:
      1. For each frequency band, compute the channel covariance matrix
      2. Treat it as a density matrix (normalize trace to 1)
      3. Compute quantum-inspired metrics:
         - Von Neumann entropy: measures "entanglement" between channels
         - Purity: how mixed/pure the state is
         - Participation ratio: effective dimensionality

    High entropy = channels are highly coupled (entangled)
    Low entropy = channels are independent

    This extends the coffeine covariance approach with quantum
    information-theoretic measures.
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    n_channels = len(ch_names)

    features = {}

    for band_name, (fmin, fmax) in bands.items():
        sos = sig.butter(4, [fmin, fmax], btype="band", fs=sfreq, output="sos")

        # Compute covariance across epochs
        covs = []
        for epoch in data:
            filtered = np.array([sig.sosfilt(sos, epoch[ch]) for ch in range(n_channels)])
            cov = np.cov(filtered)
            covs.append(cov)
        mean_cov = np.mean(covs, axis=0)

        # Normalize to density matrix (trace = 1)
        trace = np.trace(mean_cov)
        if trace > 0:
            rho = mean_cov / trace
        else:
            rho = np.eye(n_channels) / n_channels

        # Eigendecomposition
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # remove numerical zeros

        # Von Neumann entropy: S = -Tr(rho * log(rho))
        vn_entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))
        features[f"tn_vn_entropy_{band_name}"] = vn_entropy

        # Purity: Tr(rho^2) — 1/n for maximally mixed, 1 for pure state
        purity = np.sum(eigenvalues ** 2)
        features[f"tn_purity_{band_name}"] = purity

        # Participation ratio: 1 / Tr(rho^2) — effective number of dimensions
        pr = 1.0 / (purity + 1e-10)
        features[f"tn_participation_ratio_{band_name}"] = pr

        # Entanglement between frontal and parietal subgroups
        frontal_idx = [i for i, ch in enumerate(ch_names)
                       if ch in ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"]]
        parietal_idx = [i for i, ch in enumerate(ch_names)
                        if ch in ["P3", "Pz", "P4", "O1", "O2"]]

        if frontal_idx and parietal_idx:
            # Reduced density matrix for frontal subsystem
            rho_frontal = rho[np.ix_(frontal_idx, frontal_idx)]
            trace_f = np.trace(rho_frontal)
            if trace_f > 0:
                rho_frontal = rho_frontal / trace_f
            eigs_f = np.linalg.eigvalsh(rho_frontal)
            eigs_f = eigs_f[eigs_f > 1e-10]
            entropy_frontal = -np.sum(eigs_f * np.log2(eigs_f + 1e-15))
            features[f"tn_entropy_frontal_{band_name}"] = entropy_frontal

            # Reduced density matrix for parietal subsystem
            rho_parietal = rho[np.ix_(parietal_idx, parietal_idx)]
            trace_p = np.trace(rho_parietal)
            if trace_p > 0:
                rho_parietal = rho_parietal / trace_p
            eigs_p = np.linalg.eigvalsh(rho_parietal)
            eigs_p = eigs_p[eigs_p > 1e-10]
            entropy_parietal = -np.sum(eigs_p * np.log2(eigs_p + 1e-15))
            features[f"tn_entropy_parietal_{band_name}"] = entropy_parietal

            # Mutual information: S(frontal) + S(parietal) - S(total)
            # Positive = frontal and parietal are correlated
            mi = entropy_frontal + entropy_parietal - vn_entropy
            features[f"tn_mutual_info_fp_{band_name}"] = mi

    # Cross-band features: how does the density matrix structure change
    # between bands? Non-commutativity test.
    band_list = list(bands.keys())
    for i in range(len(band_list)):
        for j in range(i + 1, len(band_list)):
            b1, b2 = band_list[i], band_list[j]

            key1 = f"tn_vn_entropy_{b1}"
            key2 = f"tn_vn_entropy_{b2}"
            if key1 in features and key2 in features:
                # Entropy ratio between bands
                features[f"tn_entropy_ratio_{b1}_{b2}"] = (
                    features[key1] / (features[key2] + 1e-10)
                )

    return features


# ──────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────

def extract_quantum_features(epochs_dict, config):
    """
    Extract all quantum-inspired features for all subjects.

    Returns DataFrame with quantum features (one row per subject).
    """
    bands = config["features"]["bands"]

    all_features = []

    # Group epochs by subject
    subjects = {}
    for (sub, cond), epochs in epochs_dict.items():
        if sub not in subjects:
            subjects[sub] = {}
        subjects[sub][cond] = epochs

    for sub_id in sorted(subjects.keys()):
        print(f"  Quantum features: {sub_id}")
        sub_epochs = subjects[sub_id]

        # Use Eyes_Open as primary condition
        primary_cond = "Eyes_Open"
        if primary_cond not in sub_epochs:
            primary_cond = list(sub_epochs.keys())[0]

        epochs = sub_epochs[primary_cond]
        row = {"subject_id": sub_id}

        # 1. QEPP
        try:
            qepp = compute_qepp_features(epochs, bands)
            row.update(qepp)
            print(f"    QEPP: {len(qepp)} features")
        except Exception as e:
            print(f"    QEPP ERROR: {e}")

        # 2. Quantum probability interactions
        try:
            qi = compute_quantum_probability_features(epochs, bands)
            row.update(qi)
            print(f"    Quantum probability: {len(qi)} features")
        except Exception as e:
            print(f"    QI ERROR: {e}")

        # 3. Tensor network features
        try:
            tn = compute_tensor_features(epochs, bands)
            row.update(tn)
            print(f"    Tensor network: {len(tn)} features")
        except Exception as e:
            print(f"    TN ERROR: {e}")

        n_quantum = len(row) - 1
        print(f"    Total quantum features: {n_quantum}")
        all_features.append(row)

    return pd.DataFrame(all_features)


def run_quantum_exploration(config, epochs_dict, full_df):
    """
    Full quantum-inspired exploration:
    1. Extract quantum features
    2. Merge with behavioral data
    3. Compare quantum vs classical classification
    4. Test whether quantum features add predictive value
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif

    print("\n" + "=" * 60)
    print("EXPLORATORY: Quantum-Inspired Feature Engineering")
    print("=" * 60)

    # Extract quantum features
    quantum_df = extract_quantum_features(epochs_dict, config)

    # Merge with behavioral labels
    target_col = "ef_group_global"
    labels = full_df[["subject_id", target_col]].dropna()
    quantum_merged = quantum_df.merge(labels, on="subject_id", how="inner")

    print(f"\nQuantum features shape: {quantum_df.shape}")
    print(f"Merged with labels: {len(quantum_merged)} subjects")

    # Feature columns
    q_cols = [c for c in quantum_merged.columns
              if c not in ["subject_id", target_col]]
    print(f"Quantum feature count: {len(q_cols)}")

    if len(quantum_merged) < 10 or len(q_cols) == 0:
        print("Not enough data for quantum classification comparison")
        return quantum_df

    y = quantum_merged[target_col].astype(int)
    X_quantum = quantum_merged[q_cols].fillna(0)

    # Also get classical features for comparison
    classical_cols = [c for c in full_df.columns
                      if c.startswith(("psd_", "tbr_", "faa_", "coh_", "alpha_reactivity"))]
    X_classical = full_df.set_index("subject_id").loc[
        quantum_merged["subject_id"], classical_cols
    ].fillna(0).values

    # Combined: classical + quantum
    X_combined = np.hstack([X_classical, X_quantum.values])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    n_select = min(10, len(q_cols))

    feature_sets = {
        "classical_only": X_classical,
        "quantum_only": X_quantum.values,
        "classical_plus_quantum": X_combined,
    }

    print("\n--- Classification Comparison ---")
    print(f"{'Feature Set':35s} | {'Model':20s} | {'Bal. Accuracy':15s} | {'AUC':8s}")
    print("-" * 85)

    # Build all models matching Stage 4 (proposal Section 2.4.2)
    models_to_test = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, C=0.1, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)),
        ("SVM", __import__("sklearn.svm", fromlist=["SVC"]).SVC(
            kernel="rbf", C=1.0, probability=True, random_state=42)),
        ("KNN", __import__("sklearn.neighbors", fromlist=["KNeighborsClassifier"]).KNeighborsClassifier(
            n_neighbors=5)),
        ("MLP", __import__("sklearn.neural_network", fromlist=["MLPClassifier"]).MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42)),
    ]

    # Optional: gradient boosting models
    try:
        from xgboost import XGBClassifier
        models_to_test.append(("XGBoost", XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42, use_label_encoder=False,
            eval_metric="logloss", verbosity=0)))
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier
        models_to_test.append(("LightGBM", LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            num_leaves=8, min_child_samples=5,
            random_state=42, verbose=-1)))
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier as _CatBoost
        from sklearn.base import BaseEstimator, ClassifierMixin

        class _CatBoostWrap(BaseEstimator, ClassifierMixin):
            _estimator_type = "classifier"
            def __init__(self): pass
            def fit(self, X, y):
                self._m = _CatBoost(iterations=100, depth=3, learning_rate=0.1,
                                    random_seed=42, verbose=0)
                self._m.fit(X, y)
                self.classes_ = np.array(sorted(set(y)))
                return self
            def predict(self, X): return self._m.predict(X).flatten().astype(int)
            def predict_proba(self, X): return self._m.predict_proba(X)
            def __sklearn_tags__(self):
                tags = super().__sklearn_tags__()
                tags.estimator_type = "classifier"
                return tags

        models_to_test.append(("CatBoost", _CatBoostWrap()))
    except ImportError:
        pass

    results = []
    for fs_name, X in feature_sets.items():
        n_sel = min(10, X.shape[1])
        for model_name, model in models_to_test:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=n_sel)),
                ("clf", model),
            ])

            try:
                from sklearn.metrics import make_scorer, recall_score
                scoring = {
                    "balanced_accuracy": "balanced_accuracy",
                    "roc_auc": "roc_auc",
                    "sensitivity": make_scorer(recall_score, pos_label=1),
                    "specificity": make_scorer(recall_score, pos_label=0),
                }
                scores = cross_validate(
                    pipe, X, y, cv=cv,
                    scoring=scoring,
                    error_score="raise",
                )
                acc = np.mean(scores["test_balanced_accuracy"])
                auc = np.mean(scores["test_roc_auc"])
                std = np.std(scores["test_balanced_accuracy"])
                sens = np.mean(scores["test_sensitivity"])
                spec = np.mean(scores["test_specificity"])

                results.append({
                    "feature_set": fs_name,
                    "model": model_name,
                    "balanced_accuracy": round(acc, 3),
                    "std": round(std, 3),
                    "auc": round(auc, 3),
                    "sensitivity": round(sens, 3),
                    "specificity": round(spec, 3),
                })

                print(f"{fs_name:35s} | {model_name:20s} | "
                      f"{acc:.3f} (+/-{std:.3f})   | {auc:.3f}")
            except Exception as e:
                print(f"{fs_name:35s} | {model_name:20s} | ERROR: {e}")

    # Save results
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    quantum_df.to_csv(os.path.join(output_dir, "quantum_features.csv"), index=False)
    pd.DataFrame(results).to_csv(
        os.path.join(output_dir, "quantum_vs_classical.csv"), index=False
    )

    # Summary
    if results:
        res_df = pd.DataFrame(results)
        best_classical = res_df[res_df["feature_set"] == "classical_only"]["balanced_accuracy"].max()
        best_quantum = res_df[res_df["feature_set"] == "quantum_only"]["balanced_accuracy"].max()
        best_combined = res_df[res_df["feature_set"] == "classical_plus_quantum"]["balanced_accuracy"].max()

        print(f"\n--- Summary ---")
        print(f"  Best classical:  {best_classical:.3f}")
        print(f"  Best quantum:    {best_quantum:.3f}")
        print(f"  Best combined:   {best_combined:.3f}")

        delta = best_combined - best_classical
        print(f"  Quantum lift:    {delta:+.3f} "
              f"({'quantum adds value' if delta > 0.02 else 'no clear benefit'})")

    print(f"\nQuantum features saved: {output_dir}/quantum_features.csv")
    print(f"Comparison saved: {output_dir}/quantum_vs_classical.csv")

    return quantum_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.io import load_config
    from stages.stage1_cleaning import load_cleaned_epochs

    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    config = load_config()

    # Load pre-computed epochs and full dataset
    import mne
    epoch_dir = os.path.join(config["paths"]["output_dir"], "cleaned_epochs")
    all_epochs = {}
    for f in sorted(os.listdir(epoch_dir)):
        if f.endswith("-epo.fif"):
            parts = f.replace("-epo.fif", "").split("_", 1)
            epochs = mne.read_epochs(os.path.join(epoch_dir, f), verbose=False)
            all_epochs[(parts[0], parts[1])] = epochs

    full_df = pd.read_csv(os.path.join(config["paths"]["output_dir"], "full_dataset.csv"))

    run_quantum_exploration(config, all_epochs, full_df)
