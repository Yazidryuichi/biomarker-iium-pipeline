"""
Stage 3: Feature Building (primitives)
======================================
Reads latest preprocessing output, extracts raw primitive features per subject
per condition. Composites (TBR, FAA, a_priori) live in Stage 4.

Primitives extracted:
  PSD          absolute + relative band power per channel (np.trapezoid)
  Coherence    per pair per band, per-epoch then averaged (no concatenation)
  Wavelet      multi-scale CWT (sampled to keep dimensionality reasonable)
  Hjorth       activity / mobility / complexity per channel
  Entropy      spectral entropy per channel
  PAC          theta-phase x beta-amplitude modulation index per channel
  Covariance   per-band covariance matrix (vectorized triu)

Optional (params.aperiodic_correction):
  psd_periodic_<band>_<ch>  band power from specparam periodic component,
                            removing the 1/f aperiodic background. Fix for the
                            age-confound critique (pediatric 1/f flattens with age).

Output: output/<ts>/{features.csv, cov_matrices.npz, run_notes.json}
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml
from scipy import signal as sig
from scipy.stats import entropy as sp_entropy

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

STAGE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = STAGE_DIR.parent
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")
COND_PREFIX = {"Eyes_Open": "eo", "Eyes_Closed": "ec"}


def load_config():
    with open(STAGE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def make_output_dir():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = STAGE_DIR / "output" / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=REPO_ROOT,
        ).decode().strip()
    except Exception:
        return "unknown"


def latest_output(root):
    root = resolve(root)
    if not root.is_dir():
        return None
    runs = sorted([d for d in root.iterdir()
                   if d.is_dir() and TS_RE.match(d.name)], reverse=True)
    return runs[0] if runs else None


def load_epochs(preproc_dir, conditions):
    """Load all <sid>_<cond>-epo.fif under <preproc_dir>/cleaned_epochs/."""
    epoch_dir = preproc_dir / "cleaned_epochs"
    if not epoch_dir.is_dir():
        raise FileNotFoundError(f"missing cleaned_epochs/ in {preproc_dir}")
    conds_sorted = sorted(set(conditions), key=len, reverse=True)
    out = {}
    for f in sorted(epoch_dir.glob("*-epo.fif")):
        base = f.stem.removesuffix("-epo")
        sid, cond = None, None
        for c in conds_sorted:
            if base.endswith(f"_{c}"):
                cond = c; sid = base[:-(len(c) + 1)]; break
        if cond is None:
            print(f"  WARN: skip unmatched filename {f.name}")
            continue
        out[(sid, cond)] = mne.read_epochs(str(f), verbose=False)
    return out


# ──────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────

def extract_psd(epochs, bands):
    """Absolute + relative band power per channel via Welch + np.trapezoid."""
    psd = epochs.compute_psd(method="welch", fmin=0.5, fmax=45, verbose=False)
    data = psd.get_data()                # (n_ep, n_ch, n_freqs)
    freqs = psd.freqs
    mean_psd = np.mean(data, axis=0)     # (n_ch, n_freqs)
    total = np.trapezoid(mean_psd, freqs, axis=1)
    out = {}
    ch_names = epochs.ch_names
    for band, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        bp = np.trapezoid(mean_psd[:, mask], freqs[mask], axis=1)
        for i, ch in enumerate(ch_names):
            out[f"psd_abs_{band}_{ch}"] = float(bp[i])
            out[f"psd_rel_{band}_{ch}"] = float(bp[i] / total[i]) if total[i] > 0 else 0.0
    return out, mean_psd, freqs


def extract_psd_periodic(mean_psd, freqs, ch_names, bands, sp_cfg):
    """Re-integrate band power on the periodic (1/f-removed) PSD via specparam."""
    try:
        from specparam import SpectralModel
    except Exception:
        try:
            from fooof import FOOOF as SpectralModel  # legacy name
        except Exception:
            return {"_specparam_available": False}

    fmin, fmax = sp_cfg["freq_range"]
    mask = (freqs >= fmin) & (freqs <= fmax)
    f_use = freqs[mask]
    out = {"_specparam_available": True}
    for i, ch in enumerate(ch_names):
        try:
            m = SpectralModel(
                peak_width_limits=sp_cfg["peak_width_limits"],
                max_n_peaks=int(sp_cfg["max_n_peaks"]),
                min_peak_height=float(sp_cfg["min_peak_height"]),
                aperiodic_mode="fixed",
                verbose=False,
            )
            m.fit(f_use, mean_psd[i, mask])
            # periodic = full fit - aperiodic in log space; back to linear
            log_full = m.fooofed_spectrum_         # log10 power
            log_ap = m._ap_fit
            log_per = log_full - log_ap
            per = np.power(10.0, log_per)
        except Exception:
            per = np.full_like(f_use, np.nan)
        for band, (lo, hi) in bands.items():
            bmask = (f_use >= lo) & (f_use < hi)
            if bmask.any() and np.all(np.isfinite(per[bmask])):
                bp = float(np.trapezoid(per[bmask], f_use[bmask]))
            else:
                bp = float("nan")
            out[f"psd_periodic_{band}_{ch}"] = bp
    return out


def extract_coherence(epochs, pairs, bands):
    """Per-epoch Welch coherence, averaged across epochs."""
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = list(epochs.ch_names)
    out = {}
    for c1, c2 in pairs:
        if c1 not in ch_names or c2 not in ch_names:
            continue
        i1, i2 = ch_names.index(c1), ch_names.index(c2)
        cohs = []
        for ep in range(data.shape[0]):
            f, c = sig.coherence(data[ep, i1], data[ep, i2], fs=sfreq,
                                 nperseg=int(sfreq), noverlap=int(sfreq * 0.5))
            cohs.append(c)
        mean = np.mean(cohs, axis=0)
        for band, (lo, hi) in bands.items():
            m = (f >= lo) & (f < hi)
            if m.any():
                out[f"coh_{band}_{c1}_{c2}"] = float(np.mean(mean[m]))
    return out


def extract_wavelet(epochs, cfg):
    """Sampled CWT power per channel."""
    import pywt
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    fr = cfg["wavelet_freq_range"]
    n = int(cfg["wavelet_n_freqs"])
    freqs = np.logspace(np.log10(fr[0]), np.log10(fr[1]), n)
    scales = sfreq / (freqs * 2)
    wavelet = cfg.get("wavelet", "cmor1.5-1.0")
    pow_every = int(cfg.get("wavelet_subsample_every", 5))
    rat_every = int(cfg.get("wavelet_ratio_subsample", 10))
    out = {}
    for i, ch in enumerate(ch_names):
        ps = []
        for ep in data:
            coeffs, _ = pywt.cwt(ep[i], scales, wavelet, sampling_period=1.0 / sfreq)
            ps.append(np.abs(coeffs) ** 2)
        mp = np.mean(ps, axis=0)             # (n_freqs, n_times)
        mp_freq = np.mean(mp, axis=1)
        tv = np.var(mp, axis=1)
        mid = mp.shape[1] // 2
        early = np.mean(mp[:, :mid], axis=1)
        late = np.mean(mp[:, mid:], axis=1)
        ratio = np.log1p(late) - np.log1p(early)
        for fi in range(0, n, pow_every):
            out[f"cwt_pow_{ch}_f{fi}"] = float(mp_freq[fi])
            out[f"cwt_var_{ch}_f{fi}"] = float(tv[fi])
        for fi in range(0, n, rat_every):
            out[f"cwt_ratio_{ch}_f{fi}"] = float(ratio[fi])
    return out


def extract_hjorth(epochs):
    data = epochs.get_data()
    ch_names = epochs.ch_names
    out = {}
    for i, ch in enumerate(ch_names):
        acts, mobs, comps = [], [], []
        for ep in range(data.shape[0]):
            x = data[ep, i]
            dx = np.diff(x); ddx = np.diff(dx)
            a = np.var(x)
            mob = np.sqrt(np.var(dx) / (a + 1e-10))
            comp = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-10)) / (mob + 1e-10)
            acts.append(a); mobs.append(mob); comps.append(comp)
        out[f"hjorth_activity_{ch}"] = float(np.mean(acts))
        out[f"hjorth_mobility_{ch}"] = float(np.mean(mobs))
        out[f"hjorth_complexity_{ch}"] = float(np.mean(comps))
    return out


def extract_entropy(epochs):
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    out = {}
    for i, ch in enumerate(ch_names):
        psds = [sig.welch(ep[i], fs=sfreq, nperseg=int(sfreq))[1] for ep in data]
        mp = np.mean(psds, axis=0)
        p = mp / (np.sum(mp) + 1e-10)
        out[f"spectral_entropy_{ch}"] = float(sp_entropy(p))
    return out


def extract_pac(epochs):
    """Tort modulation index: theta phase x beta amplitude. sosfiltfilt zero-phase."""
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names
    sos_theta = sig.butter(4, [4, 8], btype="band", fs=sfreq, output="sos")
    sos_beta = sig.butter(4, [13, 30], btype="band", fs=sfreq, output="sos")
    n_bins = 18
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    out = {}
    for i, ch in enumerate(ch_names):
        mis = []
        for ep in data:
            x = ep[i]
            theta = sig.sosfiltfilt(sos_theta, x)
            beta = sig.sosfiltfilt(sos_beta, x)
            ph = np.angle(sig.hilbert(theta))
            am = np.abs(sig.hilbert(beta))
            mean_amp = np.zeros(n_bins)
            for b in range(n_bins):
                m = (ph >= phase_bins[b]) & (ph < phase_bins[b + 1])
                if m.any():
                    mean_amp[b] = am[m].mean()
            if mean_amp.sum() > 0:
                p = mean_amp / mean_amp.sum()
                p = p[p > 0]
                mi = (np.log(n_bins) + (p * np.log(p)).sum()) / np.log(n_bins)
            else:
                mi = 0.0
            mis.append(mi)
        out[f"pac_theta_beta_{ch}"] = float(np.mean(mis))
    return out


def extract_covariance(epochs, bands):
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    n_ch = data.shape[1]
    ch_names = epochs.ch_names
    out = {}
    cov_mats = {}
    triu = np.triu_indices(n_ch)
    for band, (lo, hi) in bands.items():
        sos = sig.butter(4, [lo, hi], btype="band", fs=sfreq, output="sos")
        filtered = np.array([[sig.sosfiltfilt(sos, ep[ch]) for ch in range(n_ch)]
                             for ep in data])
        covs = [np.cov(ep) for ep in filtered]
        mean_cov = np.mean(covs, axis=0)
        cov_mats[band] = mean_cov
        for r, c in zip(*triu):
            out[f"cov_{band}_{ch_names[r]}_{ch_names[c]}"] = float(mean_cov[r, c])
    return out, cov_mats


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    p = cfg["params"]
    bands = p["bands"]
    pairs = [tuple(x) for x in p["coherence_pairs"]]

    preproc_dir = latest_output(cfg["paths"]["preprocessing_root"])
    if preproc_dir is None:
        raise FileNotFoundError(
            f"No preprocessing output under {cfg['paths']['preprocessing_root']}. "
            f"Run `python preprocessing/main.py` first.")
    print(f"Reading from: {preproc_dir}")

    out_dir = make_output_dir()
    print(f"Output: {out_dir}")

    # Detect conditions from filenames
    conditions = ["Eyes_Open", "Eyes_Closed"]
    epochs_dict = load_epochs(preproc_dir, conditions)
    print(f"Loaded {len(epochs_dict)} epoch files")

    by_subject = {}
    for (sid, cond), ep in epochs_dict.items():
        by_subject.setdefault(sid, {})[cond] = ep

    aperiodic = bool(p.get("aperiodic_correction", False))
    sp_cfg = {
        "freq_range": p.get("specparam_freq_range", [1, 40]),
        "peak_width_limits": p.get("specparam_peak_width_limits", [1.0, 8.0]),
        "max_n_peaks": p.get("specparam_max_n_peaks", 6),
        "min_peak_height": p.get("specparam_min_peak_height", 0.1),
    }
    specparam_used = None

    rows = []
    cov_storage = {}
    for sid in sorted(by_subject):
        print(f"\n[{sid}]")
        row = {"subject_id": sid}
        for cond, prefix in COND_PREFIX.items():
            if cond not in by_subject[sid]:
                continue
            ep = by_subject[sid][cond]
            feats = {}

            psd_feats, mean_psd, freqs = extract_psd(ep, bands)
            feats.update(psd_feats)

            if aperiodic:
                per_feats = extract_psd_periodic(mean_psd, freqs, ep.ch_names,
                                                 bands, sp_cfg)
                if specparam_used is None:
                    specparam_used = bool(per_feats.pop("_specparam_available"))
                else:
                    per_feats.pop("_specparam_available", None)
                feats.update(per_feats)

            feats.update(extract_coherence(ep, pairs, bands))
            feats.update(extract_wavelet(ep, p))
            feats.update(extract_hjorth(ep))
            feats.update(extract_entropy(ep))
            feats.update(extract_pac(ep))
            cov_feats, cov_mats = extract_covariance(ep, bands)
            feats.update(cov_feats)
            for band, mat in cov_mats.items():
                cov_storage[f"{sid}_{prefix}_{band}"] = mat

            row.update({f"{prefix}_{k}": v for k, v in feats.items()})
            print(f"  {cond}: {len(feats)} features")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "features.csv", index=False)
    np.savez(out_dir / "cov_matrices.npz", **cov_storage)
    print(f"\nFeatures shape: {df.shape}")

    notes = {
        "stage": "feature_building",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "preprocessing_consumed": str(preproc_dir),
        "n_subjects": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "aperiodic_correction_requested": aperiodic,
        "specparam_available": specparam_used if aperiodic else None,
        "outputs": ["features.csv", "cov_matrices.npz"],
    }
    with open(out_dir / "run_notes.json", "w") as f:
        json.dump(notes, f, indent=2, default=str)
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
