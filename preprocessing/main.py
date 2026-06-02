"""
Stage 1: Preprocessing
======================
EDF -> cleaned epochs. Self-contained, no shared utils.

Flow per recording:
  1. Read EDF, set 10-20 montage, pick EEG channels.
  2. Resample to recording.sfreq, bandpass + notch filter, crop filter edges.
  3. Detect bad channels (variance MAD-z, low max-corr, flatline). Fp1/Fp2 exempt
     from variance flagging (pediatric blinks are naturally high-variance).
  4. Apply average reference (commits to avg-ref for the rest of the pipeline so
     ICA fit/apply and ICLabel classification share a reference frame).
  5. Fit ICA (infomax extended) on a 1 Hz HP copy; ICLabel classifies components;
     apply unmixing to the 0.5 Hz HP raw.
  6. Interpolate bad channels AFTER ICA; re-apply avg ref.
  7. Make fixed-length epochs; AutoReject local. No lenient retry.
  8. Drop subject-condition if surviving epochs < min_epochs floor.

Output: output/<YYYY-MM-DD_HHMMSS>/{cleaned_epochs/, qc.json, run_notes.json}
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("WARNING")

STAGE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = STAGE_DIR.parent
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")


def load_config():
    with open(STAGE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def make_output_dir():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = STAGE_DIR / "output" / ts
    (out / "cleaned_epochs").mkdir(parents=True, exist_ok=True)
    return out


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=REPO_ROOT,
        ).decode().strip()
    except Exception:
        return "unknown"


def discover_subjects(edf_dir):
    """Returns {subject_id: {condition: filepath}}."""
    edf_dir = Path(edf_dir)
    if not edf_dir.is_absolute():
        edf_dir = (REPO_ROOT / edf_dir).resolve()
    subjects = {}
    for d in sorted(edf_dir.iterdir()):
        if not (d.is_dir() and d.name.startswith("D")):
            continue
        sid = d.name
        subjects[sid] = {}
        for f in sorted(d.glob("*.edf")):
            m = re.search(r"[Ii][Gg][Ss]_(.+)$", f.stem)
            if not m:
                continue
            cond = re.sub(r"^2_IGS_", "", m.group(1))
            subjects[sid][cond] = str(f)
    return subjects


def read_edf(filepath, channels):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    mapping = {}
    for ch in raw.ch_names:
        if ch in channels:
            mapping[ch] = "eeg"
        elif ch == "EDF Annotations":
            mapping[ch] = "stim"
        else:
            mapping[ch] = "misc"
    raw.set_channel_types(mapping)
    raw.pick(mne.pick_types(raw.info, eeg=True))
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                    on_missing="warn")
    return raw


def apply_filters(raw, bandpass, notch):
    raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], verbose=False)
    if notch:
        freqs = [notch] if isinstance(notch, (int, float)) else notch
        raw.notch_filter(freqs=freqs, verbose=False)
    return raw


def detect_bad_channels(raw, p):
    """Three methods: variance MAD-z, max abs correlation, flatline std."""
    data = raw.get_data()
    ch_names = raw.ch_names
    var_thresh = float(p["bad_channel_threshold"])
    corr_thresh = float(p["bad_channel_corr_threshold"])
    flatline_std = float(p["bad_channel_flatline_std"])
    protect = set(p.get("variance_protect_channels", []))

    variances = np.var(data, axis=1)
    med = np.median(variances)
    mad = np.median(np.abs(variances - med))
    bad_var = []
    for i, v in enumerate(variances):
        if ch_names[i] in protect:
            continue
        z = 0.6745 * (v - med) / (mad + 1e-20)
        if abs(z) > var_thresh:
            bad_var.append(ch_names[i])

    corr = np.corrcoef(data)
    np.fill_diagonal(corr, 0)
    max_corrs = np.max(np.abs(corr), axis=1)
    bad_corr = [ch_names[i] for i, mc in enumerate(max_corrs) if mc < corr_thresh]

    stds = np.std(data, axis=1)
    bad_flat = [ch_names[i] for i, s in enumerate(stds) if s < flatline_std]

    return {
        "all": sorted(set(bad_var) | set(bad_corr) | set(bad_flat)),
        "by_variance": bad_var,
        "by_correlation": bad_corr,
        "by_flatline": bad_flat,
    }


def run_ica(raw, p, random_state):
    """ICA fit/apply on avg-referenced raw; ICLabel classification."""
    from mne_icalabel import label_components

    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    n_components = max(4, min(int(p.get("ica_n_components", 14)), n_eeg - 1))

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=p.get("ica_method", "infomax"),
        fit_params={"extended": bool(p.get("ica_extended", True))},
        random_state=random_state,
        max_iter="auto",
    )
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica.fit(raw_for_ica, verbose=False)

    ic = label_components(raw_for_ica, ica, method="iclabel")
    labels = list(ic["labels"])
    probs = [float(x) for x in ic["y_pred_proba"]]
    threshold = float(p.get("iclabel_threshold", 0.7))
    exclude_set = set(p.get("iclabel_exclude_labels", []))

    excl_idx, excl_lbl, excl_prob = [], [], []
    for i, (lbl, pr) in enumerate(zip(labels, probs)):
        if lbl in exclude_set and pr > threshold:
            excl_idx.append(i); excl_lbl.append(lbl); excl_prob.append(pr)
    ica.exclude = excl_idx
    raw_clean = ica.apply(raw.copy(), verbose=False)
    return raw_clean, {
        "n_components_fit": n_components,
        "excluded_idx": excl_idx,
        "excluded_labels": excl_lbl,
        "excluded_probabilities": excl_prob,
        "all_labels": labels,
        "all_probabilities": probs,
    }


def reject_epochs(epochs, p, random_state):
    """AutoReject local, fallback to fixed P2P threshold on error."""
    n_before = len(epochs)
    use_ar = bool(p.get("use_autoreject_local", True))
    fallback_uv = float(p.get("fallback_reject_uv", 150))
    max_pct = float(p.get("max_reject_pct", 0.30))

    if use_ar and n_before > 0:
        try:
            from autoreject import AutoReject
            ar = AutoReject(
                n_interpolate=[1, 2, 3],
                consensus=[0.2, 0.3, 0.4],
                cv=5, random_state=random_state, n_jobs=-1, verbose=False,
            )
            epochs_clean, log = ar.fit_transform(epochs, return_log=True)
            n_dropped = int(log.bad_epochs.sum())
            pct = n_dropped / n_before
            kept = ~log.bad_epochs
            n_interp = (float(np.mean((log.labels[kept] == 1).sum(axis=1)))
                        if kept.any() else 0.0)
            thr_uv = np.array(list(ar.threshes_.values())) * 1e6
            stats = {
                "autoreject_used": True,
                "threshold_uv_mean": float(thr_uv.mean()),
                "threshold_uv_median": float(np.median(thr_uv)),
                "n_channels_interpolated_per_epoch_mean": n_interp,
            }
            if pct > max_pct:
                print(f"    WARN: AutoReject dropped {pct:.0%} (>{max_pct:.0%}); no retry.")
            return epochs_clean, stats, n_dropped, pct
        except Exception as e:
            print(f"    WARN: AutoReject failed ({type(e).__name__}); fallback {fallback_uv} uV")

    epochs_clean = epochs.copy()
    epochs_clean.drop_bad(reject=dict(eeg=fallback_uv * 1e-6), verbose=False)
    n_dropped = n_before - len(epochs_clean)
    pct = n_dropped / n_before if n_before else 0.0
    return epochs_clean, {
        "autoreject_used": False,
        "fallback_reject_uv": fallback_uv,
    }, n_dropped, pct


def clean_one(filepath, cfg, subject, condition):
    p = cfg["params"]
    qc = {"subject": subject, "condition": condition, "filepath": filepath}

    raw = read_edf(filepath, cfg["recording"]["channels"])
    qc["duration_sec"] = float(raw.times[-1])
    qc["sfreq_raw"] = float(raw.info["sfreq"])
    qc["n_channels_raw"] = len(raw.ch_names)

    target_sfreq = float(cfg["recording"]["sfreq"])
    if raw.info["sfreq"] != target_sfreq:
        qc["resampled_from"] = float(raw.info["sfreq"])
        raw.resample(target_sfreq, verbose=False)

    apply_filters(raw, p["bandpass"], p.get("notch"))
    edge = float(p.get("filter_edge_crop_sec", 5.0))
    if raw.times[-1] > 2 * edge + 1:
        raw.crop(tmin=edge, tmax=raw.times[-1] - edge)
    qc["duration_after_crop_sec"] = float(raw.times[-1])

    bads = detect_bad_channels(raw, p)
    qc["bad_channels"] = bads["all"]
    qc["bad_by_variance"] = bads["by_variance"]
    qc["bad_by_correlation"] = bads["by_correlation"]
    qc["bad_by_flatline"] = bads["by_flatline"]
    if bads["all"]:
        raw.info["bads"] = bads["all"]

    qc["source_reference"] = "A1-A2 (Mitsar implicit)"
    qc["output_reference"] = "average"
    raw.set_eeg_reference("average", projection=False, verbose=False)

    raw_clean, ic_info = run_ica(raw, p, cfg["random_state"])
    qc["ica_n_components_fit"] = ic_info["n_components_fit"]
    qc["ica_excluded_idx"] = ic_info["excluded_idx"]
    qc["ica_excluded_labels"] = ic_info["excluded_labels"]
    qc["ica_excluded_probs"] = ic_info["excluded_probabilities"]
    qc["ica_all_labels"] = ic_info["all_labels"]

    if bads["all"]:
        raw_clean.interpolate_bads(verbose=False)
        raw_clean.set_eeg_reference("average", projection=False, verbose=False)

    duration = float(p["epoch_duration"])
    overlap = float(p.get("epoch_overlap", 0.0))
    events = mne.make_fixed_length_events(raw_clean, duration=duration, overlap=overlap)
    epochs = mne.Epochs(raw_clean, events, tmin=0,
                        tmax=duration - 1.0 / raw_clean.info["sfreq"],
                        baseline=None, preload=True, verbose=False)
    qc["n_epochs_before_reject"] = len(epochs)

    epochs_clean, rej_stats, n_dropped, pct = reject_epochs(epochs, p, cfg["random_state"])
    qc["n_epochs_after_reject"] = len(epochs_clean)
    qc["n_epochs_dropped"] = n_dropped
    qc["pct_epochs_dropped"] = round(float(pct), 4)
    qc.update(rej_stats)

    if len(epochs_clean) > 0:
        d = epochs_clean.get_data()
        qc["mean_amplitude_uv"] = float(np.mean(np.abs(d)) * 1e6)
        qc["std_amplitude_uv"] = float(np.std(d) * 1e6)
    else:
        qc["mean_amplitude_uv"] = 0.0
        qc["std_amplitude_uv"] = 0.0

    floor = int(p.get("min_epochs", 60))
    qc["status"] = "LOW_EPOCH_COUNT" if len(epochs_clean) < floor else "OK"
    return epochs_clean, qc


def main():
    cfg = load_config()
    out_dir = make_output_dir()
    epoch_dir = out_dir / "cleaned_epochs"
    print(f"Preprocessing output: {out_dir}")

    np.random.seed(cfg["random_state"])

    subjects = discover_subjects(cfg["paths"]["edf_dir"])
    print(f"Discovered {len(subjects)} subjects")

    conditions = cfg["recording"]["conditions"]
    total = sum(1 for s in subjects for c in conditions if c in subjects[s])
    qc_report = []
    processed = 0
    floor = int(cfg["params"]["min_epochs"])

    for sid in sorted(subjects):
        for cond in conditions:
            if cond not in subjects[sid]:
                print(f"  SKIP {sid}/{cond}: no file")
                continue
            processed += 1
            print(f"\n[{processed}/{total}] {sid} / {cond}")
            try:
                epochs, qc = clean_one(subjects[sid][cond], cfg, sid, cond)
                if qc["status"] == "OK":
                    epochs.save(epoch_dir / f"{sid}_{cond}-epo.fif",
                                overwrite=True, verbose=False)
                    print(f"  OK: {qc['n_epochs_after_reject']} epochs "
                          f"({qc['pct_epochs_dropped']:.0%} rejected), "
                          f"{len(qc['bad_channels'])} bad ch, "
                          f"{len(qc['ica_excluded_idx'])} ICA excl")
                else:
                    print(f"  DROP: {qc['n_epochs_after_reject']} < {floor} floor")
            except Exception as e:
                qc = {"subject": sid, "condition": cond,
                      "status": f"ERROR: {type(e).__name__}: {str(e)[:200]}"}
                print(f"  ERROR: {e}")
            qc_report.append(qc)

    with open(out_dir / "qc.json", "w") as f:
        json.dump(qc_report, f, indent=2, default=str)

    n_ok = sum(1 for q in qc_report if q.get("status") == "OK")
    n_low = sum(1 for q in qc_report if q.get("status") == "LOW_EPOCH_COUNT")
    n_err = sum(1 for q in qc_report if "ERROR" in str(q.get("status", "")))

    notes = {
        "stage": "preprocessing",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "n_subjects_discovered": len(subjects),
        "conditions": list(conditions),
        "n_files_processed": len(qc_report),
        "n_ok": n_ok, "n_low_epoch": n_low, "n_errors": n_err,
        "min_epochs_floor": floor,
        "outputs": ["cleaned_epochs/", "qc.json"],
    }
    with open(out_dir / "run_notes.json", "w") as f:
        json.dump(notes, f, indent=2, default=str)

    print(f"\nSummary: {n_ok} OK, {n_low} low-epoch dropped, {n_err} errors")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
