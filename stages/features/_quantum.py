"""
Quantum-Inspired Feature Extractors (helper module)
====================================================

Pure feature-extraction functions for QEEG data. Used by stages/features.py
when ``features.include_quantum: true`` in config. Three extractor families:

1. QEPP (Quantum Entangled Particles Pattern)
   - Treats EEG channel pairs as coupled quantum systems
   - Computes interference patterns between channels
   - Reference: Alotaibi et al. (2026), Scientific Reports

2. Quantum Probability Feature Interactions
   - Models EEG band-power combinations using quantum probability
   - Tests for interference terms (non-classical interactions)
   - Reference: Busemeyer & Bruza (2012)

3. Tensor Network Decomposition
   - Represents multi-channel EEG as a density matrix
   - Computes von Neumann entropy and entanglement-like features
   - Captures inter-channel dependencies standard methods miss

These features are EXPLORATORY. Compare against classical baselines.

References:
  Alotaibi et al. (2026). Quantum inspired feature engineering
    for EEG signal classification. Scientific Reports, 16.
  Busemeyer & Bruza (2012). Quantum Models of Cognition and
    Decision. Cambridge University Press.
  Khrennikov & Yamada (2025). Quantum-like representation of
    neuronal networks' activity. Front. Hum. Neurosci., 19.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import signal as sig

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────
# 1. QEPP: Quantum Entangled Particles Pattern
# ──────────────────────────────────────────────────────────────────

def compute_qepp_features(epochs, bands):
    """
    Quantum Entangled Particles Pattern (QEPP).
    Computes interference between channel pairs per band as the deviation
    of |state_i + state_j|^2 from |state_i|^2 + |state_j|^2.
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    n_channels = len(ch_names)

    features = {}

    for band_name, (fmin, fmax) in bands.items():
        sos = sig.butter(4, [fmin, fmax], btype="band", fs=sfreq, output="sos")
        pair_interference = {}

        for epoch in data:
            filtered = np.array([sig.sosfilt(sos, epoch[ch]) for ch in range(n_channels)])
            analytic = np.array([sig.hilbert(filtered[ch]) for ch in range(n_channels)])

            norms = np.abs(analytic)
            norms[norms == 0] = 1e-10
            states = analytic / norms

            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    pair_key = f"{ch_names[i]}_{ch_names[j]}"
                    joint = np.abs(states[i] + states[j]) ** 2
                    classical = np.abs(states[i]) ** 2 + np.abs(states[j]) ** 2
                    interference = joint - classical

                    if pair_key not in pair_interference:
                        pair_interference[pair_key] = {
                            "mean": [], "std": [], "max": []
                        }
                    pair_interference[pair_key]["mean"].append(np.mean(interference))
                    pair_interference[pair_key]["std"].append(np.std(interference))
                    pair_interference[pair_key]["max"].append(np.max(np.abs(interference)))

        # Keep only key pairs to limit dimensionality
        key_pairs = [
            "Fz_Cz", "Fz_Pz", "F3_P3", "F4_P4",
            "F3_F4", "C3_C4", "P3_P4",
            "Fp1_Fz", "Fp2_Fz",
            "Fz_O1", "Fz_O2",
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

def compute_quantum_probability_features(epochs, bands, n_bins=10):
    """
    Interference-term features between band-power pairs at frontal/central channels.
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    n_channels = len(ch_names)

    features = {}

    band_powers = {}
    for band_name, (fmin, fmax) in bands.items():
        sos = sig.butter(4, [fmin, fmax], btype="band", fs=sfreq, output="sos")
        powers = []
        for epoch in data:
            filtered = np.array([sig.sosfilt(sos, epoch[ch]) for ch in range(n_channels)])
            powers.append(np.mean(filtered ** 2, axis=1))
        band_powers[band_name] = np.array(powers)

    band_pairs = [
        ("theta", "beta"),
        ("theta", "alpha"),
        ("alpha", "beta"),
        ("delta", "beta"),
    ]

    for band_a, band_b in band_pairs:
        if band_a not in band_powers or band_b not in band_powers:
            continue

        for ch_idx, ch in enumerate(ch_names):
            if ch not in ["Fz", "F3", "F4", "Cz", "C3", "C4"]:
                continue

            a = band_powers[band_a][:, ch_idx]
            b = band_powers[band_b][:, ch_idx]

            if len(a) < 10:
                continue

            a_bins = np.clip(
                np.digitize(a, np.linspace(a.min(), a.max(), n_bins + 1)[1:-1]),
                0, n_bins - 1
            )
            b_bins = np.clip(
                np.digitize(b, np.linspace(b.min(), b.max(), n_bins + 1)[1:-1]),
                0, n_bins - 1
            )

            p_a = np.bincount(a_bins, minlength=n_bins) / len(a_bins)
            p_b = np.bincount(b_bins, minlength=n_bins) / len(b_bins)

            joint_counts = np.zeros((n_bins, n_bins))
            for ai, bi in zip(a_bins, b_bins):
                joint_counts[ai, bi] += 1
            p_joint = joint_counts / joint_counts.sum() if joint_counts.sum() > 0 else joint_counts

            p_classical = np.outer(p_a, p_b)
            interference = p_joint - p_classical

            features[f"qi_mean_{band_a}_{band_b}_{ch}"] = np.mean(np.abs(interference))
            features[f"qi_max_{band_a}_{band_b}_{ch}"] = np.max(np.abs(interference))
            features[f"qi_frob_{band_a}_{band_b}_{ch}"] = np.linalg.norm(interference, "fro")
            features[f"qi_total_{band_a}_{band_b}_{ch}"] = np.sum(np.abs(interference))

    return features


# ──────────────────────────────────────────────────────────────────
# 3. Tensor Network / Density-Matrix Features
# ──────────────────────────────────────────────────────────────────

def compute_tensor_features(epochs, bands):
    """
    Density-matrix-based features: von Neumann entropy, purity, participation
    ratio, and submatrix entropy coupling between frontal/parietal subgroups.
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    n_channels = len(ch_names)

    features = {}

    for band_name, (fmin, fmax) in bands.items():
        sos = sig.butter(4, [fmin, fmax], btype="band", fs=sfreq, output="sos")

        covs = []
        for epoch in data:
            filtered = np.array([sig.sosfilt(sos, epoch[ch]) for ch in range(n_channels)])
            covs.append(np.cov(filtered))
        mean_cov = np.mean(covs, axis=0)

        trace = np.trace(mean_cov)
        if trace > 0:
            rho = mean_cov / trace
        else:
            rho = np.eye(n_channels) / n_channels

        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        vn_entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))
        features[f"tn_vn_entropy_{band_name}"] = vn_entropy

        purity = np.sum(eigenvalues ** 2)
        features[f"tn_purity_{band_name}"] = purity

        pr = 1.0 / (purity + 1e-10)
        features[f"tn_participation_ratio_{band_name}"] = pr

        frontal_idx = [i for i, ch in enumerate(ch_names)
                       if ch in ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"]]
        parietal_idx = [i for i, ch in enumerate(ch_names)
                        if ch in ["P3", "Pz", "P4", "O1", "O2"]]

        if frontal_idx and parietal_idx:
            rho_frontal = rho[np.ix_(frontal_idx, frontal_idx)]
            trace_f = np.trace(rho_frontal)
            if trace_f > 0:
                rho_frontal = rho_frontal / trace_f
            eigs_f = np.linalg.eigvalsh(rho_frontal)
            eigs_f = eigs_f[eigs_f > 1e-10]
            entropy_frontal = -np.sum(eigs_f * np.log2(eigs_f + 1e-15))
            features[f"tn_entropy_frontal_{band_name}"] = entropy_frontal

            rho_parietal = rho[np.ix_(parietal_idx, parietal_idx)]
            trace_p = np.trace(rho_parietal)
            if trace_p > 0:
                rho_parietal = rho_parietal / trace_p
            eigs_p = np.linalg.eigvalsh(rho_parietal)
            eigs_p = eigs_p[eigs_p > 1e-10]
            entropy_parietal = -np.sum(eigs_p * np.log2(eigs_p + 1e-15))
            features[f"tn_entropy_parietal_{band_name}"] = entropy_parietal

            # Note: this is NOT proper quantum mutual information (which
            # requires partial trace over a tensor product structure).
            # The covariance submatrix is not a partial trace. Named
            # "entropy_coupling" to avoid confusion.
            entropy_coupling = entropy_frontal + entropy_parietal - vn_entropy
            features[f"tn_entropy_coupling_fp_{band_name}"] = entropy_coupling

    band_list = list(bands.keys())
    for i in range(len(band_list)):
        for j in range(i + 1, len(band_list)):
            b1, b2 = band_list[i], band_list[j]
            key1 = f"tn_vn_entropy_{b1}"
            key2 = f"tn_vn_entropy_{b2}"
            if key1 in features and key2 in features:
                features[f"tn_entropy_ratio_{b1}_{b2}"] = (
                    features[key1] / (features[key2] + 1e-10)
                )

    return features


# ──────────────────────────────────────────────────────────────────
# Per-subject orchestrator
# ──────────────────────────────────────────────────────────────────

def extract_quantum_features_per_subject(epochs, bands):
    """
    Run QEPP + quantum-probability + tensor-network extractors on a single
    subject's epochs. Returns dict of feature_name -> value.
    """
    feats = {}
    try:
        feats.update(compute_qepp_features(epochs, bands))
    except Exception as e:
        print(f"    QEPP ERROR: {e}")
    try:
        feats.update(compute_quantum_probability_features(epochs, bands))
    except Exception as e:
        print(f"    QI ERROR: {e}")
    try:
        feats.update(compute_tensor_features(epochs, bands))
    except Exception as e:
        print(f"    TN ERROR: {e}")
    return feats


def compute_qi_with_nbins(epochs, bands, n_bins):
    """
    Helper for binning sensitivity analysis (used by the standalone diagnostic
    script). Reduced version of compute_quantum_probability_features that only
    computes the mean-interference statistic per band-pair × channel.
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    n_channels = len(ch_names)
    features = {}

    band_powers = {}
    for band_name, (fmin, fmax) in bands.items():
        sos = sig.butter(4, [fmin, fmax], btype="band", fs=sfreq, output="sos")
        powers = []
        for epoch in data:
            filtered = np.array([sig.sosfilt(sos, epoch[ch]) for ch in range(n_channels)])
            powers.append(np.mean(filtered ** 2, axis=1))
        band_powers[band_name] = np.array(powers)

    band_pairs = [("theta", "beta"), ("theta", "alpha")]
    for band_a, band_b in band_pairs:
        if band_a not in band_powers or band_b not in band_powers:
            continue
        for ch_idx, ch in enumerate(ch_names):
            if ch not in ["Fz", "F3", "F4", "Cz"]:
                continue
            a = band_powers[band_a][:, ch_idx]
            b = band_powers[band_b][:, ch_idx]
            if len(a) < 10:
                continue

            a_bins = np.clip(np.digitize(a, np.linspace(a.min(), a.max(), n_bins + 1)[1:-1]), 0, n_bins - 1)
            b_bins = np.clip(np.digitize(b, np.linspace(b.min(), b.max(), n_bins + 1)[1:-1]), 0, n_bins - 1)

            p_a = np.bincount(a_bins, minlength=n_bins) / len(a_bins)
            p_b = np.bincount(b_bins, minlength=n_bins) / len(b_bins)

            joint_counts = np.zeros((n_bins, n_bins))
            for ai, bi in zip(a_bins, b_bins):
                joint_counts[ai, bi] += 1
            p_joint = joint_counts / joint_counts.sum() if joint_counts.sum() > 0 else joint_counts

            interference = p_joint - np.outer(p_a, p_b)
            features[f"qi_mean_{band_a}_{band_b}_{ch}"] = np.mean(np.abs(interference))

    return features
