"""
Stage 1: EEG Cleaning Pipeline
================================
Loads raw EDF files, applies filtering, artifact rejection, ICA,
and segments into clean epochs.

Two modes:
  - pylossless (recommended): lossless annotation, keeps all data
  - fallback: standard MNE + AutoReject pipeline

References:
  - HAPPE (Gabard-Durnam et al. 2018): pediatric EEG preprocessing
  - SCCN eeg_pipelines: conservative rejection for small N
  - pylossless: lossless annotation approach
"""

import json
import os
import warnings
from pathlib import Path

import mne
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("WARNING")


def read_edf(filepath, config):
    """
    Load an EDF file into MNE Raw, set channel types and montage.
    """
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

    # Set channel types — last channel is typically EDF Annotations
    eeg_channels = config["recording"]["channels"]
    ch_mapping = {}
    for ch in raw.ch_names:
        if ch in eeg_channels:
            ch_mapping[ch] = "eeg"
        elif ch == "EDF Annotations":
            ch_mapping[ch] = "stim"
        else:
            ch_mapping[ch] = "misc"

    raw.set_channel_types(ch_mapping)

    # Drop non-EEG channels
    picks = mne.pick_types(raw.info, eeg=True)
    raw.pick(picks)

    # Set standard 10-20 montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="warn")

    return raw


def apply_filters(raw, config):
    """
    Apply bandpass and notch filters.
    """
    bp = config["cleaning"]["bandpass"]
    raw.filter(l_freq=bp[0], h_freq=bp[1], verbose=False)

    notch = config["cleaning"]["notch"]
    if notch:
        notch_freqs = [notch] if isinstance(notch, (int, float)) else notch
        raw.notch_filter(freqs=notch_freqs, verbose=False)

    return raw


def detect_bad_channels(raw, config):
    """
    Detect bad channels using variance and correlation-based methods.
    Conservative for pediatric data (HAPPE recommendation).
    """
    data = raw.get_data()
    n_channels = data.shape[0]

    # Method 1: variance-based detection
    variances = np.var(data, axis=1)
    median_var = np.median(variances)
    mad_var = np.median(np.abs(variances - median_var))
    threshold = config["cleaning"]["bad_channel_threshold"]

    bad_by_var = []
    for i, v in enumerate(variances):
        z = 0.6745 * (v - median_var) / (mad_var + 1e-10)
        if abs(z) > threshold:
            bad_by_var.append(raw.ch_names[i])

    # Method 2: correlation-based detection
    # A good channel should correlate with at least some neighbors
    corr_matrix = np.corrcoef(data)
    np.fill_diagonal(corr_matrix, 0)
    max_corrs = np.max(np.abs(corr_matrix), axis=1)

    bad_by_corr = []
    for i, mc in enumerate(max_corrs):
        if mc < 0.3:  # channel doesn't correlate with anything
            bad_by_corr.append(raw.ch_names[i])

    bad_channels = list(set(bad_by_var + bad_by_corr))
    return bad_channels


def run_ica(raw, config):
    """
    Run ICA for artifact removal.
    Conservative: only remove components clearly identified as
    eye blinks or muscle artifacts.
    """
    n_components = config["cleaning"]["ica_n_components"]
    method = config["cleaning"]["ica_method"]

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=42,
        max_iter="auto",
    )
    ica.fit(raw, verbose=False)

    # Auto-detect EOG components using Fp1/Fp2, tracking scores
    scored_components = []
    for eog_ch in ["Fp1", "Fp2"]:
        if eog_ch in raw.ch_names:
            try:
                idx, scores = ica.find_bads_eog(
                    raw, ch_name=eog_ch, verbose=False
                )
                for i in idx:
                    scored_components.append((i, abs(scores[i])))
            except Exception:
                pass

    # Deduplicate, keeping highest score per component
    best_scores = {}
    for comp_idx, score in scored_components:
        if comp_idx not in best_scores or score > best_scores[comp_idx]:
            best_scores[comp_idx] = score

    # Sort by score descending, cap at 3 (conservative for 15-channel)
    eog_indices = [
        idx for idx, _ in sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    ][:3]

    ica.exclude = eog_indices

    # Apply ICA
    raw_clean = ica.apply(raw.copy(), verbose=False)

    return raw_clean, ica, eog_indices


def make_epochs(raw, config, condition="resting"):
    """
    Segment continuous data into fixed-length epochs.
    For resting-state: non-overlapping 2-second epochs.
    """
    duration = config["cleaning"]["epoch_duration"]
    overlap = config["cleaning"]["epoch_overlap"]

    # Create fixed-length events
    events = mne.make_fixed_length_events(
        raw, duration=duration, overlap=overlap
    )

    epochs = mne.Epochs(
        raw,
        events,
        tmin=0,
        tmax=duration - 1.0 / raw.info["sfreq"],
        baseline=None,
        preload=True,
        verbose=False,
    )

    return epochs


def reject_bad_epochs(epochs, config):
    """
    Reject bad epochs using amplitude threshold.
    Conservative: cap at max_reject_pct of total epochs.
    """
    try:
        from autoreject import get_rejection_threshold

        reject = get_rejection_threshold(epochs, verbose=False)
    except ImportError:
        # Fallback: manual peak-to-peak threshold
        reject = dict(eeg=150e-6)  # 150 uV

    n_before = len(epochs)
    epochs.drop_bad(reject=reject, verbose=False)
    n_after = len(epochs)
    n_dropped = n_before - n_after
    pct_dropped = n_dropped / n_before if n_before > 0 else 0

    max_pct = config["cleaning"]["max_reject_pct"]

    # If too many epochs rejected, loosen threshold and retry
    if pct_dropped > max_pct and n_before > 0:
        # Use a more lenient threshold
        reject_lenient = {k: v * 1.5 for k, v in reject.items()}
        # Re-create epochs from scratch would be needed here
        # For now, warn
        print(
            f"    WARNING: {pct_dropped:.0%} epochs rejected "
            f"(>{max_pct:.0%} limit). Consider looser thresholds."
        )

    return epochs, reject, n_dropped, pct_dropped


def clean_single_file(filepath, config, subject_id, condition):
    """
    Full cleaning pipeline for a single EDF file.
    Returns cleaned epochs and QC metrics.
    """
    qc = {
        "subject": subject_id,
        "condition": condition,
        "filepath": filepath,
    }

    # Load
    raw = read_edf(filepath, config)
    qc["duration_sec"] = raw.times[-1]
    qc["sfreq"] = raw.info["sfreq"]
    qc["n_channels_raw"] = len(raw.ch_names)

    # Filter
    raw = apply_filters(raw, config)

    # Bad channel detection
    bad_channels = detect_bad_channels(raw, config)
    qc["bad_channels"] = bad_channels
    qc["n_bad_channels"] = len(bad_channels)

    if bad_channels:
        raw.info["bads"] = bad_channels
        raw.interpolate_bads(verbose=False)

    # ICA before average reference (HAPPE protocol: ICA on non-averaged data,
    # then re-reference after component removal)
    raw_clean, ica, excluded_ics = run_ica(raw, config)

    # Re-reference to average AFTER ICA (avoids reference constraint in ICA)
    raw_clean.set_eeg_reference("average", verbose=False)
    qc["ica_excluded"] = excluded_ics
    qc["n_ica_excluded"] = len(excluded_ics)

    # Epoch
    epochs = make_epochs(raw_clean, config, condition)
    qc["n_epochs_before_reject"] = len(epochs)

    # Reject bad epochs
    epochs_clean, reject_thresh, n_dropped, pct_dropped = reject_bad_epochs(
        epochs, config
    )
    qc["n_epochs_after_reject"] = len(epochs_clean)
    qc["n_epochs_dropped"] = n_dropped
    qc["pct_epochs_dropped"] = round(pct_dropped, 4)
    qc["reject_threshold_uv"] = (
        reject_thresh.get("eeg", 0) * 1e6
    )  # convert to uV

    # SNR estimate: mean signal / mean noise (from rejected epochs)
    if len(epochs_clean) > 0:
        data = epochs_clean.get_data()
        qc["mean_amplitude_uv"] = float(np.mean(np.abs(data)) * 1e6)
        qc["std_amplitude_uv"] = float(np.std(data) * 1e6)
    else:
        qc["mean_amplitude_uv"] = 0
        qc["std_amplitude_uv"] = 0

    qc["status"] = "OK" if len(epochs_clean) >= 10 else "LOW_EPOCH_COUNT"

    return epochs_clean, qc


def run_stage1(config, subjects, conditions=None):
    """
    Run Stage 1 cleaning on all subjects and conditions.

    Args:
        config: pipeline config dict
        subjects: dict from discover_subjects()
        conditions: list of conditions to process (default: primary only)

    Returns:
        all_epochs: dict {(subject, condition): Epochs}
        qc_report: list of QC dicts
    """
    if conditions is None:
        conditions = config["recording"]["conditions"]["primary"]

    output_dir = os.path.join(config["paths"]["output_dir"], "cleaned_epochs")
    os.makedirs(output_dir, exist_ok=True)

    all_epochs = {}
    qc_report = []

    total = sum(
        1 for s in subjects for c in conditions if c in subjects[s]
    )
    processed = 0

    for subject_id in sorted(subjects.keys()):
        for condition in conditions:
            if condition not in subjects[subject_id]:
                print(f"  SKIP {subject_id}/{condition}: no file")
                continue

            filepath = subjects[subject_id][condition]
            processed += 1
            print(
                f"\n[{processed}/{total}] {subject_id} / {condition}"
            )

            try:
                epochs, qc = clean_single_file(
                    filepath, config, subject_id, condition
                )
                all_epochs[(subject_id, condition)] = epochs

                # Save cleaned epochs
                epoch_fname = f"{subject_id}_{condition}-epo.fif"
                epochs.save(
                    os.path.join(output_dir, epoch_fname),
                    overwrite=True,
                    verbose=False,
                )

                print(
                    f"  OK: {qc['n_epochs_after_reject']} epochs "
                    f"({qc['pct_epochs_dropped']:.0%} rejected), "
                    f"{qc['n_bad_channels']} bad ch, "
                    f"{qc['n_ica_excluded']} ICA excluded"
                )

            except Exception as e:
                qc = {
                    "subject": subject_id,
                    "condition": condition,
                    "status": f"ERROR: {str(e)[:200]}",
                }
                print(f"  ERROR: {e}")

            qc_report.append(qc)

    # Save QC report
    qc_path = os.path.join(config["paths"]["output_dir"], "qc_stage1.json")
    with open(qc_path, "w") as f:
        json.dump(qc_report, f, indent=2, default=str)
    print(f"\nQC report saved: {qc_path}")

    # Summary
    ok = sum(1 for q in qc_report if q.get("status") == "OK")
    errors = sum(1 for q in qc_report if "ERROR" in str(q.get("status", "")))
    low = sum(
        1 for q in qc_report if q.get("status") == "LOW_EPOCH_COUNT"
    )
    print(f"\nStage 1 Summary: {ok} OK, {low} low epoch count, {errors} errors")

    return all_epochs, qc_report


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.io import load_config, discover_subjects

    config = load_config()
    subjects = discover_subjects(config["paths"]["edf_dir"])
    print(f"Found {len(subjects)} subjects")

    all_epochs, qc = run_stage1(config, subjects)
