"""
Stage 2: QEEG Feature Extraction (raw primitives only)
=======================================================
Extracts three tiers of *primitive* features from cleaned epochs.
Math-derived composites (TBR, FAA, alpha reactivity) live in Stage 3
(feature engineering), which computes them from the band powers stored
here.

  2A. Conventional QEEG: PSD per band per channel, coherence per pair
  2B. Advanced: wavelet-Fourier, PAC, entropy, Hjorth parameters
  2C. Covariance (coffeine): frequency-band covariance matrices

Output: features.csv (one row per subject)
"""

import logging
import os
import warnings

import mne
import numpy as np
import pandas as pd
import pywt
from scipy import signal as sig
from scipy.stats import entropy as sp_entropy

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

logger = logging.getLogger("biomarker_iium.stage2")


# ──────────────────────────────────────────────────────────────────
# 2A. Conventional QEEG Features
# ──────────────────────────────────────────────────────────────────

def extract_psd_features(epochs, bands, method="welch"):
    """
    Power Spectral Density: absolute and relative power per band per channel.
    """
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    ch_names = epochs.ch_names
    n_channels = len(ch_names)

    features = {}

    # Compute PSD using Welch's method (averaged across epochs)
    psd = epochs.compute_psd(method="welch", fmin=0.5, fmax=45, verbose=False)
    psd_data = psd.get_data()  # (n_epochs, n_channels, n_freqs)
    freqs = psd.freqs

    # Average across epochs
    mean_psd = np.mean(psd_data, axis=0)  # (n_channels, n_freqs)

    # Total power via numerical integration (V², resolution-independent)
    total_power = np.trapz(mean_psd, freqs, axis=1)  # (n_channels,)

    for band_name, (fmin, fmax) in bands.items():
        freq_mask = (freqs >= fmin) & (freqs < fmax)
        band_power = np.trapz(mean_psd[:, freq_mask], freqs[freq_mask], axis=1)

        for i, ch in enumerate(ch_names):
            features[f"psd_abs_{band_name}_{ch}"] = band_power[i]
            features[f"psd_rel_{band_name}_{ch}"] = (
                band_power[i] / total_power[i] if total_power[i] > 0 else 0
            )

    return features


def extract_coherence(epochs, coherence_pairs, bands):
    """
    Coherence between channel pairs in each frequency band.

    Uses per-epoch coherence estimation averaged across epochs,
    instead of concatenating epochs (which creates discontinuity
    artifacts at epoch boundaries).
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = list(epochs.ch_names)
    features = {}

    for ch1, ch2 in coherence_pairs:
        if ch1 not in ch_names or ch2 not in ch_names:
            continue
        idx1 = ch_names.index(ch1)
        idx2 = ch_names.index(ch2)

        # Compute coherence per epoch, then average
        epoch_cohs = []
        for ep in range(data.shape[0]):
            f, coh = sig.coherence(
                data[ep, idx1, :], data[ep, idx2, :],
                fs=sfreq, nperseg=int(sfreq * 1.0),
                noverlap=int(sfreq * 0.5),
            )
            epoch_cohs.append(coh)

        mean_coh = np.mean(epoch_cohs, axis=0)

        for band_name, (fmin, fmax) in bands.items():
            mask = (f >= fmin) & (f < fmax)
            if mask.sum() > 0:
                features[f"coh_{band_name}_{ch1}_{ch2}"] = np.mean(
                    mean_coh[mask]
                )

    return features


# ──────────────────────────────────────────────────────────────────
# 2B. Advanced Features
# ──────────────────────────────────────────────────────────────────

def extract_wavelet_features(epochs, config):
    """
    Continuous wavelet transform: multi-scale time-frequency features.
    Per channel: mean power, temporal variance, early/late ratio at each frequency.
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names

    freq_range = config["features"]["wavelet_freq_range"]
    n_freqs = config["features"]["wavelet_n_freqs"]
    freqs = np.logspace(
        np.log10(freq_range[0]), np.log10(freq_range[1]), num=n_freqs
    )
    scales = sfreq / (freqs * 2)

    features = {}

    # Average CWT across epochs for efficiency
    for ch_idx, ch in enumerate(ch_names):
        powers = []
        for epoch in data:
            coeffs, _ = pywt.cwt(
                epoch[ch_idx], scales, "cmor1.5-1.0",
                sampling_period=1.0 / sfreq,
            )
            powers.append(np.abs(coeffs) ** 2)

        mean_power = np.mean(powers, axis=0)  # (n_freqs, n_times)

        # Feature 1: mean power per frequency
        mp = np.mean(mean_power, axis=1)
        for fi in range(0, n_freqs, 5):  # sample every 5th to reduce dimensionality
            features[f"cwt_pow_{ch}_f{fi}"] = mp[fi]

        # Feature 2: temporal variance (captures dynamics)
        tv = np.var(mean_power, axis=1)
        for fi in range(0, n_freqs, 5):
            features[f"cwt_var_{ch}_f{fi}"] = tv[fi]

        # Feature 3: early vs late ratio
        mid = mean_power.shape[1] // 2
        early = np.mean(mean_power[:, :mid], axis=1)
        late = np.mean(mean_power[:, mid:], axis=1)
        ratio = np.log1p(late) - np.log1p(early)
        for fi in range(0, n_freqs, 10):  # coarser sampling
            features[f"cwt_ratio_{ch}_f{fi}"] = ratio[fi]

    return features


def extract_hjorth(epochs):
    """
    Hjorth parameters: Activity, Mobility, Complexity per channel.
    Computed per-epoch then averaged (not on epoch-averaged signal,
    which would destroy temporal structure in resting-state data).
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    ch_names = epochs.ch_names
    features = {}

    for i, ch in enumerate(ch_names):
        activities, mobilities, complexities = [], [], []
        for ep in range(data.shape[0]):
            x = data[ep, i]
            dx = np.diff(x)
            ddx = np.diff(dx)

            act = np.var(x)
            mob = np.sqrt(np.var(dx) / (act + 1e-10))
            comp = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-10)) / (mob + 1e-10)

            activities.append(act)
            mobilities.append(mob)
            complexities.append(comp)

        features[f"hjorth_activity_{ch}"] = np.mean(activities)
        features[f"hjorth_mobility_{ch}"] = np.mean(mobilities)
        features[f"hjorth_complexity_{ch}"] = np.mean(complexities)

    return features


def extract_entropy(epochs):
    """
    Spectral entropy per channel: measures signal regularity.
    High entropy = complex/irregular, Low entropy = regular/periodic.
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    features = {}

    for i, ch in enumerate(ch_names):
        # Average PSD across epochs
        psds = []
        for epoch in data:
            f, pxx = sig.welch(epoch[i], fs=sfreq, nperseg=int(sfreq))
            psds.append(pxx)
        mean_psd = np.mean(psds, axis=0)

        # Normalize to probability distribution
        psd_norm = mean_psd / (np.sum(mean_psd) + 1e-10)
        se = sp_entropy(psd_norm)
        features[f"spectral_entropy_{ch}"] = se

    return features


def extract_pac(epochs, bands):
    """
    Simplified phase-amplitude coupling: theta phase x beta/gamma amplitude.
    Modulation index per channel.
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    features = {}

    for i, ch in enumerate(ch_names):
        mis = []
        for epoch in data:
            x = epoch[i]

            # Theta phase
            sos_theta = sig.butter(
                4, [4, 8], btype="band", fs=sfreq, output="sos"
            )
            theta = sig.sosfilt(sos_theta, x)
            theta_phase = np.angle(sig.hilbert(theta))

            # Beta amplitude
            sos_beta = sig.butter(
                4, [13, 30], btype="band", fs=sfreq, output="sos"
            )
            beta = sig.sosfilt(sos_beta, x)
            beta_amp = np.abs(sig.hilbert(beta))

            # Modulation index
            n_bins = 18
            phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            mean_amp = np.zeros(n_bins)
            for b in range(n_bins):
                mask = (theta_phase >= phase_bins[b]) & (
                    theta_phase < phase_bins[b + 1]
                )
                if mask.sum() > 0:
                    mean_amp[b] = np.mean(beta_amp[mask])

            if mean_amp.sum() > 0:
                p = mean_amp / mean_amp.sum()
                p = p[p > 0]
                mi = (np.log(n_bins) + np.sum(p * np.log(p))) / np.log(
                    n_bins
                )
            else:
                mi = 0.0
            mis.append(mi)

        features[f"pac_theta_beta_{ch}"] = np.mean(mis)

    return features


# ──────────────────────────────────────────────────────────────────
# 2C. Covariance Features (coffeine-compatible)
# ──────────────────────────────────────────────────────────────────

def extract_covariance_features(epochs, bands):
    """
    Frequency-band covariance matrices.
    Vectorized upper triangle for traditional ML.
    Raw matrices saved separately for Riemannian classifiers.
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    n_channels = data.shape[1]
    ch_names = epochs.ch_names
    features = {}
    cov_matrices = {}

    for band_name, (fmin, fmax) in bands.items():
        # Bandpass filter all epochs
        sos = sig.butter(4, [fmin, fmax], btype="band", fs=sfreq, output="sos")
        filtered = np.array(
            [[sig.sosfilt(sos, epoch[ch]) for ch in range(n_channels)] for epoch in data]
        )

        # Compute covariance per epoch, then average
        covs = []
        for ep in filtered:
            cov = np.cov(ep)
            covs.append(cov)
        mean_cov = np.mean(covs, axis=0)  # (n_channels, n_channels)

        cov_matrices[band_name] = mean_cov

        # Vectorize upper triangle
        triu_idx = np.triu_indices(n_channels)
        for j, (r, c) in enumerate(zip(*triu_idx)):
            features[f"cov_{band_name}_{ch_names[r]}_{ch_names[c]}"] = mean_cov[
                r, c
            ]

    return features, cov_matrices


# ──────────────────────────────────────────────────────────────────
# Main extraction orchestrator
# ──────────────────────────────────────────────────────────────────

def extract_all_features(epochs_dict, config):
    """
    Extract raw primitive features for all subjects.
    Composite/derived features (TBR, FAA, alpha reactivity) are computed
    later by stages/engineering.py.

    When ``features.include_quantum`` is true in config, also extracts
    quantum-inspired features (QEPP, QI, tensor-network) from the primary
    Eyes-Open epoch per subject. These are unprefixed columns
    (qepp_*, qi_*, tn_*) since they are cross-condition density-matrix
    representations.

    Args:
        epochs_dict: {(subject_id, condition): mne.Epochs}
        config: pipeline config

    Returns:
        features_df: DataFrame (one row per subject)
        cov_matrices: dict for Riemannian classifiers
    """
    bands = config["features"]["bands"]
    coherence_pairs = config["features"]["coherence_pairs"]
    include_quantum = config["features"].get("include_quantum", False)

    if include_quantum:
        from stages.features._quantum import extract_quantum_features_per_subject

    # Group epochs by subject
    subjects = {}
    for (sub, cond), epochs in epochs_dict.items():
        if sub not in subjects:
            subjects[sub] = {}
        subjects[sub][cond] = epochs

    # Condition prefix map — extend here if emotional conditions are added
    COND_PREFIX = {"Eyes_Open": "eo", "Eyes_Closed": "ec"}

    all_features = []
    all_cov_matrices = {}

    for sub_id in sorted(subjects.keys()):
        print(f"\n  Extracting features: {sub_id}")
        sub_epochs = subjects[sub_id]

        row = {"subject_id": sub_id}

        # Extract features per condition, prefix all columns with eo_ / ec_
        for cond, prefix in COND_PREFIX.items():
            if cond not in sub_epochs:
                continue
            epochs = sub_epochs[cond]
            cond_feats = {}

            # 2A: Conventional QEEG (primitives only)
            cond_feats.update(extract_psd_features(epochs, bands))
            cond_feats.update(extract_coherence(epochs, coherence_pairs, bands))

            # 2B: Advanced
            cond_feats.update(extract_wavelet_features(epochs, config))
            cond_feats.update(extract_hjorth(epochs))
            cond_feats.update(extract_entropy(epochs))
            cond_feats.update(extract_pac(epochs, bands))

            # 2C: Covariance
            cov_feats, cov_mats = extract_covariance_features(epochs, bands)
            cond_feats.update(cov_feats)
            all_cov_matrices[f"{sub_id}_{prefix}"] = cov_mats

            row.update({f"{prefix}_{k}": v for k, v in cond_feats.items()})

        # Quantum-inspired features (Eyes-Open primary) — unprefixed
        if include_quantum:
            primary_cond = "Eyes_Open" if "Eyes_Open" in sub_epochs else next(iter(sub_epochs))
            q_feats = extract_quantum_features_per_subject(sub_epochs[primary_cond], bands)
            row.update(q_feats)
            print(f"    +{len(q_feats)} quantum-inspired features")

        all_features.append(row)
        n_feats = len(row) - 1  # minus subject_id
        print(f"    {n_feats} features extracted")

    df = pd.DataFrame(all_features)
    return df, all_cov_matrices


def load_features(config, stage_dir=None):
    """
    Load saved features.csv. Auto-resolves the latest features directory
    if ``stage_dir`` is not given.
    """
    from utils.io import latest_stage_dir

    if stage_dir is None:
        stage_dir = latest_stage_dir(config, "features")
        if stage_dir is None:
            raise FileNotFoundError(
                "No features output found. Run `python pipeline.py --features` first."
            )
    path = os.path.join(stage_dir, "features.csv")
    df = pd.read_csv(path)
    print(f"  Loaded features: {df.shape} from {path}")
    return df


def run(config, epochs_dict=None):
    """
    Run Stage 2 feature extraction.

    Writes features.csv, cov_matrices.npz, and run_notes.json into
    ``config['output_dir']`` (created by ``load_stage_config('features')``).
    Auto-loads epochs from ``config['input_dir']`` (latest cleaning run)
    when ``epochs_dict`` is None.
    """
    from utils.io import write_stage_notes
    from stages.cleaning import load_cleaned_epochs

    print("\n" + "=" * 60)
    print("STAGE 2: Feature Extraction")
    print("=" * 60)

    cleaning_input = config.get("input_dir")
    if epochs_dict is None:
        if cleaning_input is None:
            raise FileNotFoundError(
                "No cleaning output found. Run `python pipeline.py --cleaning` first."
            )
        epochs_dict = load_cleaned_epochs(config, stage_dir=cleaning_input)

    features_df, cov_matrices = extract_all_features(epochs_dict, config)

    stage_dir = config["output_dir"]
    print(f"  Features output dir: {stage_dir}")

    features_path = os.path.join(stage_dir, "features.csv")
    features_df.to_csv(features_path, index=False)
    print(f"\nFeatures saved: {features_path}")
    print(f"Shape: {features_df.shape}")

    cov_path = os.path.join(stage_dir, "cov_matrices.npz")
    np.savez(cov_path, **{
        f"{sub}_{band}": mat
        for sub, bands in cov_matrices.items()
        for band, mat in bands.items()
    })
    print(f"Covariance matrices saved: {cov_path}")

    write_stage_notes(stage_dir, {
        "stage": "features",
        "input_cleaning_dir": cleaning_input,
        "n_subjects": int(features_df.shape[0]),
        "n_features": int(features_df.shape[1] - 1),
        "outputs": ["features.csv", "cov_matrices.npz"],
    })

    return features_df, cov_matrices
