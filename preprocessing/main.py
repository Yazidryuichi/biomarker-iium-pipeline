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


RENAME_OLD_TO_NEW = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

def read_edf(filepath, channels):
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

    # Mitsar/WinEEG pakai label 10-20 lama; rename ke nama modern SEBELUM
    # montage supaya tiap channel dapat posisi (ICLabel butuh posisi).
    present = {k: v for k, v in RENAME_OLD_TO_NEW.items() if k in raw.ch_names}
    if present:
        raw.rename_channels(present)

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

    # Gagal-keras: nggak ada lagi drop channel diam-diam
    if len(raw.ch_names) != len(channels):
        raise ValueError(
            f"{Path(filepath).name}: kept {len(raw.ch_names)} EEG ch, "
            f"expected {len(channels)}. "
            f"Hilang: {sorted(set(channels) - set(raw.ch_names))} | "
            f"Tak dikenal: {sorted(set(raw.ch_names) - set(channels))}")

    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                    on_missing="raise")
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
    qc["ica_all_probabilities"] = ic_info["all_probabilities"]

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


def find_emotion_onset(raw, cond):
    """Onset (sec) of the emotion's own marker inside its EDF.

    cond is like '1_Happy'; we match the emotion word ('happy') against the
    annotation descriptions (which carry boundary markers such as '2_Calm' or
    'BAD_ACQ_SKIP'). Markers are inconsistent across subjects — sometimes the
    file's own start marker is present, sometimes only the next emotion's
    boundary marker. When the own marker is absent we fall back to 0.0 (the
    file effectively begins at the emotion block).
    """
    word = cond.split("_", 1)[-1].strip().lower()
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        d = str(desc).strip().lower()
        if "bad" in d:
            continue
        if word and word in d:
            return float(onset)
    return 0.0


def emotion_block_end(raw, response_end):
    """End of the emotion block = onset of the next-emotion boundary marker
    (the last non-BAD annotation), used to place the within-file baseline at
    the block tail. Falls back to the recording end (minus a small margin)
    when no boundary marker beyond the response window is present.
    """
    onsets = [float(o) for o, d in zip(raw.annotations.onset,
                                       raw.annotations.description)
              if "bad" not in str(d).strip().lower()]
    rec_end = float(raw.times[-1])
    cands = [o for o in onsets if o > response_end + 1.0]
    return min(max(cands), rec_end) if cands else rec_end - 1.0


def _epoch_window(raw_clean, start, end, p, ip, cfg, overlap=None):
    """Crop a copy to [start, end], epoch, AutoReject. Returns (epochs, qc).

    `overlap` (seconds) defaults to the IAPS window overlap; the decoder block
    passes its own (decode.epoch_overlap) so the response/baseline windows and
    the full-block decoder windows can use different overlaps.
    """
    rec_end = float(raw_clean.times[-1])
    start = max(0.0, float(start)); end = min(float(end), rec_end)
    r = raw_clean.copy().crop(tmin=start, tmax=end)
    duration = float(p["epoch_duration"])
    overlap = float(ip.get("epoch_overlap", 1.0)) if overlap is None else float(overlap)
    events = mne.make_fixed_length_events(r, duration=duration, overlap=overlap)
    epochs = mne.Epochs(r, events, tmin=0,
                        tmax=duration - 1.0 / r.info["sfreq"],
                        baseline=None, preload=True, verbose=False)
    n_before = len(epochs)
    ep_clean, rej_stats, n_dropped, pct = reject_epochs(epochs, p,
                                                        cfg["random_state"])
    qc = {"window_start_sec": round(start, 2), "window_end_sec": round(end, 2),
          "n_epochs_before_reject": n_before,
          "n_epochs_after_reject": len(ep_clean),
          "n_epochs_dropped": n_dropped,
          "pct_epochs_dropped": round(float(pct), 4)}
    qc.update(rej_stats)
    return ep_clean, qc


def clean_one_emotion(filepath, cfg, subject, condition):
    """Clean one emotion EDF and crop TWO windows from a single ICA pass:

      response       first `window_sec` after the emotion onset (developed
                     stimulus response; transient-safe via min_window_start_sec)
      within_baseline last `window_sec` of the block (return-to-baseline tail
                     before the next-emotion marker) — the PRIMARY IAPS baseline

    Shares the resting cleaning path (read -> filter -> bad channels -> avg ref
    -> ICA on the full 60-115 s recording -> interpolate) so both windows live
    in the same reference frame as Eyes_Open (the sensitivity baseline). The two
    windows are non-overlapping by construction (blocks are >= 64 s).

    When `params.iaps.decode.enable`, a THIRD set of epochs is cropped from the
    SAME ICA pass: the whole emotion block (onset -> next-emotion marker), epoched
    at 2 s / `decode.epoch_overlap` (50% default). These full-block windows feed
    the EXPLORATORY affective decoder (within-subject 4-emotion classification);
    they are NOT used by the confirmatory feature_building/analysis path.

    Returns (response_epochs_or_None, baseline_epochs_or_None,
             decode_epochs_or_None, qc).
    """
    p = cfg["params"]
    ip = p.get("iaps", {}) or {}
    qc = {"subject": subject, "condition": condition, "filepath": filepath,
          "kind": "emotion"}

    raw = read_edf(filepath, cfg["recording"]["channels"])
    qc["duration_sec"] = float(raw.times[-1])
    qc["sfreq_raw"] = float(raw.info["sfreq"])
    qc["n_channels_raw"] = len(raw.ch_names)

    target_sfreq = float(cfg["recording"]["sfreq"])
    if raw.info["sfreq"] != target_sfreq:
        qc["resampled_from"] = float(raw.info["sfreq"])
        raw.resample(target_sfreq, verbose=False)

    onset = find_emotion_onset(raw, condition)
    qc["emotion_onset_sec"] = round(onset, 2)
    qc["emotion_onset_marker_found"] = bool(onset > 0.0)

    apply_filters(raw, p["bandpass"], p.get("notch"))

    bads = detect_bad_channels(raw, p)
    qc["bad_channels"] = bads["all"]
    if bads["all"]:
        raw.info["bads"] = bads["all"]

    qc["source_reference"] = "A1-A2 (Mitsar implicit)"
    qc["output_reference"] = "average"
    raw.set_eeg_reference("average", projection=False, verbose=False)

    raw_clean, ic_info = run_ica(raw, p, cfg["random_state"])
    qc["ica_n_components_fit"] = ic_info["n_components_fit"]
    qc["ica_excluded_idx"] = ic_info["excluded_idx"]
    qc["ica_excluded_labels"] = ic_info["excluded_labels"]
    qc["ica_all_labels"] = ic_info["all_labels"]
    qc["ica_all_probabilities"] = ic_info["all_probabilities"]

    if bads["all"]:
        raw_clean.interpolate_bads(verbose=False)
        raw_clean.set_eeg_reference("average", projection=False, verbose=False)

    window = float(ip.get("window_sec", 15.0))
    skip = float(ip.get("skip_after_onset_sec", 0.0))
    min_start = float(ip.get("min_window_start_sec", 5.0))
    floor = int(ip.get("min_epochs", 4))
    rec_end = float(raw_clean.times[-1])
    qc["min_epochs_floor"] = floor

    # ── Response window (first window_sec after onset) ──
    r_start = max(onset + skip, min_start)
    r_end = r_start + window
    if r_end > rec_end:
        if rec_end - min_start >= window:
            r_end = rec_end; r_start = r_end - window
            qc["response_window_slid_back"] = True
        else:
            qc["status"] = "WINDOW_TOO_SHORT"
            qc["window_start_sec"] = round(r_start, 2)
            qc["n_epochs_after_reject"] = 0
            qc["viable"] = False
            qc["base_viable"] = False
            qc["decode_viable"] = False
            return None, None, None, qc
    resp_ep, rqc = _epoch_window(raw_clean, r_start, r_end, p, ip, cfg)
    qc.update(rqc)                       # response stats at top level
    ok = len(resp_ep) >= floor
    qc["status"] = "OK" if ok else "LOW_EPOCH_COUNT"
    qc["viable"] = bool(ok)

    # ── Within-file baseline window (last window_sec of the block) ──
    block_end = emotion_block_end(raw_clean, r_end)
    b_end = min(block_end, rec_end)
    b_start = b_end - window
    base_ep = None
    if b_start >= r_end:                 # non-overlapping with the response
        base_ep_c, bqc = _epoch_window(raw_clean, b_start, b_end, p, ip, cfg)
        qc["base_window_start_sec"] = bqc["window_start_sec"]
        qc["base_window_end_sec"] = bqc["window_end_sec"]
        qc["base_n_epochs_after_reject"] = bqc["n_epochs_after_reject"]
        base_ok = len(base_ep_c) >= floor
        base_ep = base_ep_c if base_ok else None
        qc["base_viable"] = bool(base_ok)
    else:
        qc["base_viable"] = False
        qc["base_window_note"] = "block too short for a non-overlapping tail"

    # ── Full-block decoder window (EXPLORATORY; onset -> block end) ──
    dec_cfg = ip.get("decode", {}) or {}
    decode_ep = None
    if dec_cfg.get("enable", False):
        d_floor = int(dec_cfg.get("min_epochs", 20))
        d_overlap = float(dec_cfg.get("epoch_overlap", 1.0))
        d_start = r_start                       # same transient-safe onset floor
        d_end = max(block_end, r_end)           # whole block, not just response
        qc["decode_min_epochs_floor"] = d_floor
        if d_end - d_start >= float(p["epoch_duration"]):
            dec_ep_c, dqc = _epoch_window(raw_clean, d_start, d_end, p, ip, cfg,
                                          overlap=d_overlap)
            qc["decode_window_start_sec"] = dqc["window_start_sec"]
            qc["decode_window_end_sec"] = dqc["window_end_sec"]
            qc["decode_n_epochs_after_reject"] = dqc["n_epochs_after_reject"]
            dec_ok = len(dec_ep_c) >= d_floor
            decode_ep = dec_ep_c if dec_ok else None
            qc["decode_viable"] = bool(dec_ok)
        else:
            qc["decode_viable"] = False
            qc["decode_window_note"] = "block too short for decoder windows"

    return (resp_ep if ok else None), base_ep, decode_ep, qc


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

    # ─── IAPS emotion windows (viability + cleaned response segments) ───
    iaps_cfg = cfg["params"].get("iaps", {}) or {}
    emotion_conditions = cfg["recording"].get("emotion_conditions", []) or []
    n_emo_ok = 0
    if iaps_cfg.get("enable", False) and emotion_conditions:
        print(f"\n=== IAPS emotion windows "
              f"({iaps_cfg.get('window_sec', 15)} s, floor "
              f"{iaps_cfg.get('min_epochs', 4)} epochs) ===")
        for sid in sorted(subjects):
            for cond in emotion_conditions:
                if cond not in subjects[sid]:
                    print(f"  SKIP {sid}/{cond}: no file")
                    continue
                print(f"\n[emotion] {sid} / {cond}")
                try:
                    resp_ep, base_ep, decode_ep, qc = clean_one_emotion(
                        subjects[sid][cond], cfg, sid, cond)
                    if qc["status"] == "OK" and resp_ep is not None:
                        resp_ep.save(epoch_dir / f"{sid}_{cond}-epo.fif",
                                     overwrite=True, verbose=False)
                        n_emo_ok += 1
                        if base_ep is not None:
                            base_ep.save(
                                epoch_dir / f"{sid}_{cond}_base-epo.fif",
                                overwrite=True, verbose=False)
                        if decode_ep is not None:
                            decode_ep.save(
                                epoch_dir / f"{sid}_{cond}_decode-epo.fif",
                                overwrite=True, verbose=False)
                        print(f"  OK: response {qc['n_epochs_after_reject']} ep "
                              f"[{qc['window_start_sec']},{qc['window_end_sec']}]s "
                              f"| within-base "
                              f"{qc.get('base_n_epochs_after_reject', 0)} ep "
                              f"({'ok' if qc.get('base_viable') else 'NON-VIABLE'})"
                              f" | decode "
                              f"{qc.get('decode_n_epochs_after_reject', 0)} ep "
                              f"({'ok' if qc.get('decode_viable') else 'NON-VIABLE'})")
                    else:
                        print(f"  NON-VIABLE: {qc.get('status')} "
                              f"({qc.get('n_epochs_after_reject', 0)} epochs)")
                except Exception as e:
                    qc = {"subject": sid, "condition": cond, "kind": "emotion",
                          "viable": False, "base_viable": False,
                          "status": f"ERROR: {type(e).__name__}: {str(e)[:200]}"}
                    print(f"  ERROR: {e}")
                qc_report.append(qc)

    with open(out_dir / "qc.json", "w") as f:
        json.dump(qc_report, f, indent=2, default=str)

    rest_qc = [q for q in qc_report if q.get("kind") != "emotion"]
    emo_qc = [q for q in qc_report if q.get("kind") == "emotion"]
    n_ok = sum(1 for q in rest_qc if q.get("status") == "OK")
    n_low = sum(1 for q in rest_qc if q.get("status") == "LOW_EPOCH_COUNT")
    n_err = sum(1 for q in qc_report if "ERROR" in str(q.get("status", "")))
    n_emo_viable = sum(1 for q in emo_qc if q.get("viable"))
    n_emo_base_viable = sum(1 for q in emo_qc if q.get("base_viable"))
    n_emo_decode_viable = sum(1 for q in emo_qc if q.get("decode_viable"))

    notes = {
        "stage": "preprocessing",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "n_subjects_discovered": len(subjects),
        "conditions": list(conditions),
        "n_files_processed": len(qc_report),
        "n_ok": n_ok, "n_low_epoch": n_low, "n_errors": n_err,
        "min_epochs_floor": floor,
        "iaps_enabled": bool(iaps_cfg.get("enable", False)),
        "emotion_conditions": list(emotion_conditions),
        "n_emotion_files_processed": len(emo_qc),
        "n_emotion_viable": n_emo_viable,
        "n_emotion_within_baseline_viable": n_emo_base_viable,
        "iaps_decode_enabled": bool(
            (iaps_cfg.get("decode", {}) or {}).get("enable", False)),
        "n_emotion_decode_viable": n_emo_decode_viable,
        "outputs": ["cleaned_epochs/", "qc.json"],
    }
    with open(out_dir / "run_notes.json", "w") as f:
        json.dump(notes, f, indent=2, default=str)

    print(f"\nSummary: {n_ok} OK, {n_low} low-epoch dropped, {n_err} errors")
    if emo_qc:
        print(f"IAPS emotion: {n_emo_viable}/{len(emo_qc)} viable windows")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
