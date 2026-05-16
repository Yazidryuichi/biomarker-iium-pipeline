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
import logging
import os
import warnings
from pathlib import Path

import mne
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("WARNING")

logger = logging.getLogger("biomarker_iium.stage1")


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
    Detect bad channels via three independent methods:
      1. Variance (MAD-z outlier on per-channel variance)
      2. Correlation (max abs correlation with any other channel)
      3. Flatline (std below an absolute floor)

    All thresholds configurable via ``cleaning.params`` keys
    ``bad_channel_threshold``, ``bad_channel_corr_threshold``,
    ``bad_channel_flatline_std``. Returns a dict so callers can record
    which method flagged each channel.
    """
    data = raw.get_data()
    ch_names = raw.ch_names

    var_thresh = config["cleaning"].get("bad_channel_threshold", 3.5)
    corr_thresh = config["cleaning"].get("bad_channel_corr_threshold", 0.2)
    flatline_std = float(config["cleaning"].get("bad_channel_flatline_std", 1e-7))
    # Channels exempt from variance-based flagging (still eligible for
    # correlation/flatline). Frontal Fp1/Fp2 are naturally high-variance
    # in pediatric EEG because of blink amplitude; removing them from
    # the ICA fit hides exactly the artifact ICLabel needs to classify.
    variance_protect = set(
        config["cleaning"].get("variance_protect_channels", ["Fp1", "Fp2"])
    )

    # Variance MAD-z
    variances = np.var(data, axis=1)
    median_var = np.median(variances)
    mad_var = np.median(np.abs(variances - median_var))
    bad_by_var = []
    for i, v in enumerate(variances):
        if ch_names[i] in variance_protect:
            continue
        z = 0.6745 * (v - median_var) / (mad_var + 1e-20)
        if abs(z) > var_thresh:
            bad_by_var.append(ch_names[i])

    # Correlation (low max abs correlation = isolated channel)
    corr = np.corrcoef(data)
    np.fill_diagonal(corr, 0)
    max_corrs = np.max(np.abs(corr), axis=1)
    bad_by_corr = [
        ch_names[i] for i, mc in enumerate(max_corrs) if mc < corr_thresh
    ]

    # Flatline (per-channel std below absolute floor)
    stds = np.std(data, axis=1)
    bad_by_flat = [
        ch_names[i] for i, s in enumerate(stds) if s < flatline_std
    ]

    all_bad = sorted(set(bad_by_var) | set(bad_by_corr) | set(bad_by_flat))
    return {
        "all": all_bad,
        "by_variance": bad_by_var,
        "by_correlation": bad_by_corr,
        "by_flatline": bad_by_flat,
    }


def run_ica(raw, config):
    """
    Run ICA + ICLabel artifact-component classification.

    Caller contract: ``raw`` must already be **average-referenced** and
    have ``info['bads']`` populated. We fit ICA on a 1 Hz high-pass copy
    of that avg-ref data so the mixing matrix and ICLabel classification
    both live in the same reference frame as the data we then apply to.

    Components whose label is in ``iclabel_exclude_labels`` AND whose
    probability exceeds ``iclabel_threshold`` are removed.
    """
    from mne_icalabel import label_components

    method = config["cleaning"].get("ica_method", "infomax")
    fit_params = dict(config["cleaning"].get("ica_fit_params", {"extended": True}))
    random_state = config.get("random_state", 42)
    iclabel_threshold = float(config["cleaning"].get("iclabel_threshold", 0.8))
    exclude_labels = set(config["cleaning"].get("iclabel_exclude_labels", [
        "eye blink",
        "muscle artifact",
        "heart beat",
        "line noise",
        "channel noise",
    ]))

    # Component count: cap from config; -1 for avg-ref rank loss.
    # pick_types(eeg=True) excludes bads by default, so no separate
    # n_bads subtraction is needed.
    n_eeg_good = len(mne.pick_types(raw.info, eeg=True))
    cfg_cap = config["cleaning"].get("ica_n_components", 14)
    n_components = max(4, min(cfg_cap, n_eeg_good - 1))

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        fit_params=fit_params,
        random_state=random_state,
        max_iter="auto",
    )

    # 1 Hz HP copy of the (already avg-ref) input — slow drifts < 1 Hz
    # would degrade ICA decomposition. Reference is inherited from raw.
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica.fit(raw_for_ica, verbose=False)

    # ICLabel on the same avg-ref data the ICA was fit on
    ic_dict = label_components(raw_for_ica, ica, method="iclabel")
    labels = list(ic_dict["labels"])
    probs = [float(p) for p in ic_dict["y_pred_proba"]]

    exclude_idx = []
    excluded_labels = []
    excluded_probs = []
    for idx, (lbl, prob) in enumerate(zip(labels, probs)):
        if lbl in exclude_labels and prob > iclabel_threshold:
            exclude_idx.append(idx)
            excluded_labels.append(lbl)
            excluded_probs.append(prob)

    ica.exclude = exclude_idx
    # Apply on raw (avg-ref, 0.5 Hz HP) — same reference as the fit copy,
    # only the high-pass differs and that does not affect the unmixing W.
    raw_clean = ica.apply(raw.copy(), verbose=False)

    ic_info = {
        "excluded_idx": exclude_idx,
        "excluded_labels": excluded_labels,
        "excluded_probabilities": excluded_probs,
        "all_labels": labels,
        "all_probabilities": probs,
        "n_components_fit": n_components,
    }
    return raw_clean, ica, ic_info


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
    Reject / interpolate bad epochs via AutoReject (local).

    AutoReject finds per-channel thresholds via CV and, per epoch, either
    interpolates a small number of bad channels or drops the epoch
    entirely. No data-driven retry — if ``pct_dropped`` exceeds
    ``max_reject_pct`` we just log a warning. The ``min_epochs`` floor
    in ``clean_single_file`` is the gate that handles unrecoverable
    recordings.

    Fallback (only when AutoReject errors out, e.g. too few epochs for
    CV): fixed peak-to-peak threshold ``fallback_reject_uv`` µV.
    """
    use_ar = config["cleaning"].get("use_autoreject_local", True)
    random_state = config.get("random_state", 42)
    fallback_uv = float(config["cleaning"].get("fallback_reject_uv", 150))
    max_pct = config["cleaning"].get("max_reject_pct", 0.30)

    n_before = len(epochs)

    if use_ar and n_before > 0:
        try:
            from autoreject import AutoReject

            ar = AutoReject(
                n_interpolate=[1, 2, 3],
                consensus=[0.2, 0.3, 0.4],
                cv=5,
                random_state=random_state,
                n_jobs=-1,
                verbose=False,
            )
            epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
            n_dropped = int(reject_log.bad_epochs.sum())
            pct_dropped = n_dropped / n_before if n_before > 0 else 0.0

            kept_mask = ~reject_log.bad_epochs
            if kept_mask.any():
                n_interp_per_kept = (reject_log.labels[kept_mask] == 1).sum(axis=1)
                mean_n_interp = float(np.mean(n_interp_per_kept))
            else:
                mean_n_interp = 0.0

            thresholds_uv = np.array(list(ar.threshes_.values())) * 1e6
            stats = {
                "autoreject_used": True,
                "autoreject_threshold_uv_mean": float(thresholds_uv.mean()),
                "autoreject_threshold_uv_median": float(np.median(thresholds_uv)),
                "autoreject_threshold_uv_max": float(thresholds_uv.max()),
                "n_channels_interpolated_per_epoch_mean": mean_n_interp,
                "reject_threshold_uv": None,
            }

            if pct_dropped > max_pct:
                print(
                    f"    WARNING: AutoReject dropped {pct_dropped:.0%} "
                    f"epochs (>{max_pct:.0%} limit); no retry."
                )
            return epochs_clean, stats, n_dropped, pct_dropped
        except Exception as e:
            print(
                f"    WARNING: AutoReject failed ({type(e).__name__}: "
                f"{str(e)[:120]}); using fixed {fallback_uv:.0f} µV threshold."
            )

    # Fallback: fixed peak-to-peak threshold (no retry)
    reject = dict(eeg=fallback_uv * 1e-6)
    epochs_clean = epochs.copy()
    epochs_clean.drop_bad(reject=reject, verbose=False)
    n_dropped = n_before - len(epochs_clean)
    pct_dropped = n_dropped / n_before if n_before > 0 else 0.0

    if pct_dropped > max_pct:
        print(
            f"    WARNING: Fallback threshold dropped {pct_dropped:.0%} "
            f"epochs (>{max_pct:.0%} limit); no retry."
        )

    stats = {
        "autoreject_used": False,
        "autoreject_threshold_uv_mean": None,
        "autoreject_threshold_uv_median": None,
        "autoreject_threshold_uv_max": None,
        "n_channels_interpolated_per_epoch_mean": 0.0,
        "reject_threshold_uv": float(fallback_uv),
    }
    return epochs_clean, stats, n_dropped, pct_dropped


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

    # Enforce target sampling rate (EDF files may vary)
    target_sfreq = config["recording"]["sfreq"]
    if raw.info["sfreq"] != target_sfreq:
        original_sfreq = raw.info["sfreq"]
        raw.resample(target_sfreq, verbose=False)
        qc["resampled_from"] = original_sfreq

    # Filter
    raw = apply_filters(raw, config)

    # Trim filter edge artifacts. Default 5 s covers a 0.5 Hz HP FIR
    # transition band; configurable via cleaning.filter_edge_crop_sec.
    edge_crop = float(config["cleaning"].get("filter_edge_crop_sec", 5.0))
    if raw.times[-1] > (2 * edge_crop + 1):
        raw.crop(tmin=edge_crop, tmax=raw.times[-1] - edge_crop)
    qc["filter_edge_crop_sec"] = edge_crop
    qc["duration_after_crop_sec"] = float(raw.times[-1])

    # Bad channel detection — mark but do NOT interpolate yet
    # (HAPPE protocol: interpolate AFTER ICA, not before, because
    # interpolated channels inject smoothed data that degrades ICA)
    bad_info = detect_bad_channels(raw, config)
    qc["bad_channels"] = bad_info["all"]
    qc["n_bad_channels"] = len(bad_info["all"])
    qc["bad_by_variance"] = bad_info["by_variance"]
    qc["bad_by_correlation"] = bad_info["by_correlation"]
    qc["bad_by_flatline"] = bad_info["by_flatline"]

    if bad_info["all"]:
        raw.info["bads"] = bad_info["all"]

    # Apply average reference NOW, before ICA. The source EDFs are
    # Mitsar A1-A2 referenced (no M1/M2 in file), and ICLabel was
    # trained on average-referenced data, so we commit to avg ref for
    # the rest of the pipeline. set_eeg_reference excludes bads from
    # the average automatically.
    qc["source_reference"] = "A1-A2 (Mitsar implicit)"
    qc["output_reference"] = "average"
    raw.set_eeg_reference("average", projection=False, verbose=False)

    # ICA + ICLabel — fit/classify/apply all on the same avg-ref data
    raw_clean, ica, ic_info = run_ica(raw, config)
    qc["ica_excluded"] = ic_info["excluded_idx"]
    qc["n_ica_excluded"] = len(ic_info["excluded_idx"])
    qc["ica_labels"] = ic_info["excluded_labels"]
    qc["ica_probabilities"] = ic_info["excluded_probabilities"]
    qc["ica_all_labels"] = ic_info["all_labels"]
    qc["ica_all_probabilities"] = ic_info["all_probabilities"]
    qc["ica_n_components_fit"] = ic_info["n_components_fit"]

    # Interpolate bad channels AFTER ICA component removal
    if bad_info["all"]:
        raw_clean.interpolate_bads(verbose=False)
        # Re-apply avg ref so the newly-restored channels contribute to
        # the average (previously they were excluded from the avg calc).
        raw_clean.set_eeg_reference("average", projection=False, verbose=False)

    # Epoch
    epochs = make_epochs(raw_clean, config, condition)
    qc["n_epochs_before_reject"] = len(epochs)

    # Reject bad epochs (AutoReject local + interpolation; no retry)
    epochs_clean, reject_stats, n_dropped, pct_dropped = reject_bad_epochs(
        epochs, config
    )
    qc["n_epochs_after_reject"] = len(epochs_clean)
    qc["n_epochs_dropped"] = n_dropped
    qc["pct_epochs_dropped"] = round(pct_dropped, 4)
    qc.update(reject_stats)

    # SNR-ish summary stats
    if len(epochs_clean) > 0:
        data = epochs_clean.get_data()
        qc["mean_amplitude_uv"] = float(np.mean(np.abs(data)) * 1e6)
        qc["std_amplitude_uv"] = float(np.std(data) * 1e6)
    else:
        qc["mean_amplitude_uv"] = 0
        qc["std_amplitude_uv"] = 0

    min_epochs = config["cleaning"].get("min_epochs", 10)
    if len(epochs_clean) < min_epochs:
        qc["status"] = "LOW_EPOCH_COUNT"
    else:
        qc["status"] = "OK"

    return epochs_clean, qc


def load_cleaned_epochs(config, stage_dir=None):
    """
    Load previously saved cleaned epochs. Auto-resolves the latest
    cleaning output directory if ``stage_dir`` is not given.
    """
    from utils.io import latest_stage_dir

    if stage_dir is None:
        stage_dir = latest_stage_dir(config, "cleaning")
        if stage_dir is None:
            raise FileNotFoundError(
                "No cleaning output found. Run `python pipeline.py --cleaning` first."
            )

    epoch_dir = os.path.join(stage_dir, "cleaned_epochs")
    all_epochs = {}
    for f in sorted(os.listdir(epoch_dir)):
        if f.endswith("-epo.fif"):
            parts = f.replace("-epo.fif", "").split("_", 1)
            subject_id = parts[0]
            condition = parts[1] if len(parts) > 1 else "unknown"
            epochs = mne.read_epochs(
                os.path.join(epoch_dir, f), verbose=False
            )
            all_epochs[(subject_id, condition)] = epochs

    print(f"  Loaded {len(all_epochs)} epoch files from {epoch_dir}")
    return all_epochs


def run(config, subjects, conditions=None):
    """
    Run Stage 1 cleaning on all subjects and conditions.

    Writes cleaned_epochs/, qc.json, and run_notes.json into
    ``config['output_dir']`` (created by ``load_stage_config('cleaning')``).

    Args:
        config: merged stage config from ``load_stage_config('cleaning')``
        subjects: dict from discover_subjects()
        conditions: list of conditions to process (default: primary only)

    Returns:
        all_epochs: dict {(subject, condition): Epochs}
        qc_report: list of QC dicts
    """
    from utils.io import write_stage_notes

    if conditions is None:
        conditions = config["recording"]["conditions"]["primary"]

    stage_dir = config["output_dir"]
    output_dir = os.path.join(stage_dir, "cleaned_epochs")
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Cleaning output dir: {stage_dir}")

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

                if qc["status"] == "LOW_EPOCH_COUNT":
                    # Skip saving — file will be invisible to downstream stages
                    print(
                        f"  DROP: {qc['n_epochs_after_reject']} epochs "
                        f"(<{config['cleaning'].get('min_epochs', 10)} floor), "
                        f"{qc['pct_epochs_dropped']:.0%} rejected"
                    )
                else:
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
    qc_path = os.path.join(stage_dir, "qc.json")
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

    write_stage_notes(stage_dir, {
        "stage": "cleaning",
        "n_subjects": len(subjects),
        "conditions": list(conditions),
        "n_files_processed": len(qc_report),
        "n_ok": ok,
        "n_low_epoch": low,
        "n_errors": errors,
        "outputs": ["cleaned_epochs/", "qc.json"],
    })

    return all_epochs, qc_report


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from utils.io import load_stage_config, discover_subjects

    config = load_stage_config("cleaning")
    subjects = discover_subjects(config["paths"]["edf_dir"])
    print(f"Found {len(subjects)} subjects")

    all_epochs, qc = run(config, subjects)
