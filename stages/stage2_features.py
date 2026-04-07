"""
Stage 2: QEEG Feature Extraction
==================================
Extracts three tiers of features from cleaned epochs:
  2A. Conventional QEEG: PSD, TBR, FAA, alpha reactivity, coherence
  2B. Advanced: wavelet-Fourier, PAC, entropy, Hjorth parameters
  2C. Covariance (coffeine): frequency-band covariance matrices

Output: features.csv (one row per subject, ~200+ columns)
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


def extract_tbr(epochs, bands, tbr_channels):
    """
    Theta/Beta Ratio at specified frontal/central channels.
    """
    psd = epochs.compute_psd(method="welch", fmin=0.5, fmax=45, verbose=False)
    psd_data = np.mean(psd.get_data(), axis=0)
    freqs = psd.freqs
    ch_names = epochs.ch_names

    theta_mask = (freqs >= bands["theta"][0]) & (freqs < bands["theta"][1])
    beta_mask = (freqs >= bands["beta"][0]) & (freqs < bands["beta"][1])

    features = {}
    for ch in tbr_channels:
        if ch not in ch_names:
            continue
        idx = ch_names.index(ch)
        theta_power = np.trapz(psd_data[idx, theta_mask], freqs[theta_mask])
        beta_power = np.trapz(psd_data[idx, beta_mask], freqs[beta_mask])
        tbr = theta_power / beta_power if beta_power > 0 else np.nan
        features[f"tbr_{ch}"] = tbr

    # Mean frontal TBR
    tbr_vals = [v for v in features.values() if not np.isnan(v)]
    features["tbr_frontal_mean"] = np.mean(tbr_vals) if tbr_vals else np.nan

    return features


def extract_faa(epochs, bands, left_ch, right_ch):
    """
    Frontal Alpha Asymmetry: ln(alpha_right) - ln(alpha_left).
    Positive = greater left frontal activity (approach motivation).
    """
    psd = epochs.compute_psd(method="welch", fmin=0.5, fmax=45, verbose=False)
    psd_data = np.mean(psd.get_data(), axis=0)
    freqs = psd.freqs
    ch_names = epochs.ch_names

    alpha_mask = (freqs >= bands["alpha"][0]) & (freqs < bands["alpha"][1])

    features = {}
    if left_ch in ch_names and right_ch in ch_names:
        left_idx = ch_names.index(left_ch)
        right_idx = ch_names.index(right_ch)
        alpha_left = np.trapz(psd_data[left_idx, alpha_mask], freqs[alpha_mask])
        alpha_right = np.trapz(psd_data[right_idx, alpha_mask], freqs[alpha_mask])

        faa = np.log(alpha_right + 1e-10) - np.log(alpha_left + 1e-10)
        features["faa_F4_F3"] = faa
    else:
        features["faa_F4_F3"] = np.nan

    return features


def extract_alpha_reactivity(epochs_eo, epochs_ec, bands):
    """
    Alpha reactivity: change in alpha power from Eyes Closed to Eyes Open.
    EC > EO indicates normal desynchronization.
    """
    features = {}

    if epochs_eo is None or epochs_ec is None:
        features["alpha_reactivity_global"] = np.nan
        return features

    psd_eo = epochs_eo.compute_psd(method="welch", fmin=0.5, fmax=45, verbose=False)
    psd_ec = epochs_ec.compute_psd(method="welch", fmin=0.5, fmax=45, verbose=False)

    freqs = psd_eo.freqs
    alpha_mask = (freqs >= bands["alpha"][0]) & (freqs < bands["alpha"][1])

    mean_eo = np.mean(psd_eo.get_data(), axis=0)
    mean_ec = np.mean(psd_ec.get_data(), axis=0)

    alpha_eo = np.mean(np.trapz(mean_eo[:, alpha_mask], freqs[alpha_mask], axis=1))
    alpha_ec = np.mean(np.trapz(mean_ec[:, alpha_mask], freqs[alpha_mask], axis=1))

    # Reactivity: how much alpha drops from EC to EO
    reactivity = (alpha_ec - alpha_eo) / (alpha_ec + 1e-10)
    features["alpha_reactivity_global"] = reactivity

    # Per-channel reactivity for posterior channels
    for ch in ["O1", "O2", "Pz"]:
        ch_names = epochs_eo.ch_names
        if ch in ch_names:
            idx = ch_names.index(ch)
            eo_alpha = np.trapz(mean_eo[idx, alpha_mask], freqs[alpha_mask])
            ec_alpha = np.trapz(mean_ec[idx, alpha_mask], freqs[alpha_mask])
            r = (ec_alpha - eo_alpha) / (ec_alpha + 1e-10)
            features[f"alpha_reactivity_{ch}"] = r

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
    Extract all features for all subjects.

    Args:
        epochs_dict: {(subject_id, condition): mne.Epochs}
        config: pipeline config

    Returns:
        features_df: DataFrame (one row per subject)
        cov_matrices: dict for Riemannian classifiers
    """
    bands = config["features"]["bands"]
    tbr_channels = config["features"]["tbr_channels"]
    faa_left = config["features"]["faa_left"]
    faa_right = config["features"]["faa_right"]
    coherence_pairs = config["features"]["coherence_pairs"]

    # Group epochs by subject
    subjects = {}
    for (sub, cond), epochs in epochs_dict.items():
        if sub not in subjects:
            subjects[sub] = {}
        subjects[sub][cond] = epochs

    all_features = []
    all_cov_matrices = {}

    for sub_id in sorted(subjects.keys()):
        print(f"\n  Extracting features: {sub_id}")
        sub_epochs = subjects[sub_id]

        # Use Eyes_Open as primary condition
        primary_cond = "Eyes_Open"
        if primary_cond not in sub_epochs:
            primary_cond = list(sub_epochs.keys())[0]

        epochs = sub_epochs[primary_cond]
        row = {"subject_id": sub_id, "condition": primary_cond}

        # 2A: Conventional QEEG
        row.update(extract_psd_features(epochs, bands))
        row.update(extract_tbr(epochs, bands, tbr_channels))
        row.update(extract_faa(epochs, bands, faa_left, faa_right))
        row.update(
            extract_coherence(epochs, coherence_pairs, bands)
        )

        # Alpha reactivity (needs both EO and EC)
        eo = sub_epochs.get("Eyes_Open")
        ec = sub_epochs.get("Eyes_Closed")
        row.update(extract_alpha_reactivity(eo, ec, bands))

        # 2B: Advanced
        row.update(extract_wavelet_features(epochs, config))
        row.update(extract_hjorth(epochs))
        row.update(extract_entropy(epochs))
        row.update(extract_pac(epochs, bands))

        # 2C: Covariance
        cov_feats, cov_mats = extract_covariance_features(epochs, bands)
        row.update(cov_feats)
        all_cov_matrices[sub_id] = cov_mats

        all_features.append(row)
        n_feats = len(row) - 2  # minus subject_id and condition
        print(f"    {n_feats} features extracted")

    df = pd.DataFrame(all_features)
    return df, all_cov_matrices


def run_stage2(config, epochs_dict):
    """Run Stage 2 and save results."""
    print("\n" + "=" * 60)
    print("STAGE 2: Feature Extraction")
    print("=" * 60)

    features_df, cov_matrices = extract_all_features(epochs_dict, config)

    # Save
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    features_path = os.path.join(output_dir, "features.csv")
    features_df.to_csv(features_path, index=False)
    print(f"\nFeatures saved: {features_path}")
    print(f"Shape: {features_df.shape}")

    # Save covariance matrices
    cov_path = os.path.join(output_dir, "cov_matrices.npz")
    np.savez(cov_path, **{
        f"{sub}_{band}": mat
        for sub, bands in cov_matrices.items()
        for band, mat in bands.items()
    })
    print(f"Covariance matrices saved: {cov_path}")

    return features_df, cov_matrices
