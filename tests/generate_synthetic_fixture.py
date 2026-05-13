"""Generate a synthetic fixture matching the Biomarker_IIUM pipeline schema.

Produces 28 fake subjects with:
  - EDF files (15 channels, 250 Hz, ~30 s) of gaussian noise plus
    a faint alpha-band sine, so Stage 1 cleaning has something to do
    and Stage 2 feature extraction produces sensible PSD output.
  - AUFEI-O / Flanker / Digit Span xlsx files with valid column schemas
    and plausible value ranges.

This fixture is for end-to-end pipeline verification by reviewers / CI.
It contains no real patient data and no PHI. EDF subject names are
deterministic placeholders (`Subject01`, `Subject02`, ...).

Run:
    python tests/generate_synthetic_fixture.py \
        --out tests/fixtures/synthetic \
        --n-subjects 28 \
        --seed 42

After generation, run the pipeline against the fixture:
    python run_all.py \
        --config tests/fixtures/synthetic/config.fixture.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "C3", "Cz", "C4",
    "P3", "Pz", "P4",
    "O1", "O2",
]
SFREQ = 250
DURATION_S = 30
N_SAMPLES = SFREQ * DURATION_S


def make_signal(rng: np.random.Generator, ef_level: float) -> np.ndarray:
    """One subject's multichannel signal.

    `ef_level` in [0, 1] modulates a small alpha-band oscillation; this
    gives the classifier something to find without ground-truthing any
    specific feature. Values are in volts; MNE expects volts.
    """
    n_ch = len(CHANNELS)
    t = np.arange(N_SAMPLES) / SFREQ
    # White noise at ~20 µV RMS.
    sig = rng.standard_normal((n_ch, N_SAMPLES)) * 20e-6
    # Alpha-band carrier ~10 Hz, amplitude scaled by ef_level.
    alpha = (8e-6 * ef_level) * np.sin(2 * np.pi * 10.0 * t)
    # Posterior channels (P3, Pz, P4, O1, O2) get the alpha most.
    posterior = [CHANNELS.index(c) for c in ("P3", "Pz", "P4", "O1", "O2")]
    sig[posterior, :] += alpha[np.newaxis, :]
    # Slow drift on frontal channels (something for ICA to remove).
    frontal = [CHANNELS.index(c) for c in ("Fp1", "Fp2")]
    drift = 40e-6 * np.sin(2 * np.pi * 0.3 * t)
    sig[frontal, :] += drift[np.newaxis, :]
    # Occasional blink-like artefacts in Fp1/Fp2.
    blink_times = rng.choice(N_SAMPLES, size=4, replace=False)
    for ts in blink_times:
        a, b = max(ts - 50, 0), min(ts + 50, N_SAMPLES)
        sig[frontal, a:b] += 100e-6 * np.hanning(b - a)
    return sig


def write_edf(filepath: Path, data: np.ndarray) -> None:
    """Write a 15-channel EDF file via MNE. Imports MNE lazily so this
    module can be imported in environments without MNE for arg parsing.
    """
    import mne
    info = mne.create_info(ch_names=CHANNELS, sfreq=SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
    # MNE export_raw writes EDF via pyedflib or edflib. Suppress its info.
    mne.export.export_raw(
        str(filepath), raw, fmt="edf", overwrite=True, verbose=False,
    )


def make_aufei(subject_ids, sexes, dobs, rng):
    """AUFEI-O behavioural Excel schema (ID, Sex, DoB, WM*, IC*, CF*, P#, SF*)."""
    rows = []
    for sid, sex, dob in zip(subject_ids, sexes, dobs):
        row = {"ID": sid, "Sex": sex, "DoB": dob}
        # 5 items per domain, scored 1–5 Likert. Some subjects high-EF, some low.
        # Use last digit of subject ID as a quick high/low marker (no real info,
        # just enough variance for classification target derivation).
        ef_bias = 4 if int(sid[-1]) % 2 == 0 else 2
        for prefix, n_items in [("WM", 5), ("IC", 5), ("CF", 5), ("P", 5), ("SF", 5)]:
            for i in range(1, n_items + 1):
                row[f"{prefix}{i}"] = float(np.clip(
                    rng.normal(ef_bias, 0.8), 1.0, 5.0
                ))
        rows.append(row)
    return pd.DataFrame(rows)


def make_flanker(subject_ids, rng):
    """Flanker pilot CSV columns expected by utils.io.load_flanker."""
    rows = []
    for sid in subject_ids:
        rows.append({
            "ID": sid,
            "acc_overall": float(rng.uniform(0.7, 0.98)),
            "acc_incongruent": float(rng.uniform(0.6, 0.95)),
            "flanker_effect": float(rng.normal(40, 15)),
            "rt_mean": float(rng.normal(550, 80)),
            "rt_incongruent": float(rng.normal(610, 90)),
            "rt_congruent": float(rng.normal(530, 80)),
            "rt_cv": float(rng.uniform(0.15, 0.35)),
            "ddm_delta_v": float(rng.normal(0.3, 0.12)),
        })
    return pd.DataFrame(rows)


def make_digit_span(subject_ids, rng):
    """Digit Span: forward + backward span counts."""
    rows = []
    for sid in subject_ids:
        rows.append({
            "ID": sid,
            "DS_Forward": int(rng.integers(3, 8)),
            "DS_Backward": int(rng.integers(2, 7)),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for fixture.")
    parser.add_argument("--n-subjects", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-emotional", action="store_true",
                        help="Generate emotional condition files in addition to EO/EC.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    out = args.out
    edf_root = out / "EF_Biomarker" / "EDF_Files"
    beh_root = out / "data_root"
    edf_root.mkdir(parents=True, exist_ok=True)
    beh_root.mkdir(parents=True, exist_ok=True)
    (beh_root / "AUFEI-O").mkdir(exist_ok=True)

    primary = ["Eyes_Open", "Eyes_Closed"]
    emotional = ["1_Happy", "2_Calm", "3_Sad", "4_Scare"]
    conditions = primary + (emotional if args.include_emotional else [])

    subject_ids = [f"D{i:07d}" for i in range(args.n_subjects)]
    sexes = list(rng.choice(["M", "F"], size=args.n_subjects))
    dobs = pd.to_datetime("2018-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 5, size=args.n_subjects), unit="D"
    )

    for i, sid in enumerate(subject_ids):
        subj_dir = edf_root / sid
        subj_dir.mkdir(exist_ok=True)
        ef_level = float(rng.uniform(0.2, 0.9))
        for cond in conditions:
            data = make_signal(rng, ef_level)
            fname = f"1_{sexes[i]}_1_Subject{i + 1:02d}_IGS_{cond}.edf"
            write_edf(subj_dir / fname, data)
        if (i + 1) % 5 == 0:
            print(f"  generated {i + 1}/{args.n_subjects} subjects")

    aufei = make_aufei(subject_ids, sexes, dobs, rng)
    flanker = make_flanker(subject_ids, rng)
    digit = make_digit_span(subject_ids, rng)

    aufei.to_excel(beh_root / "AUFEI-O" / "AUFEI-O_Cleaned.xlsx", index=False)
    flanker.to_excel(beh_root / "Flanker_Test_Pilot.xlsx", index=False)
    digit.to_excel(beh_root / "Digit_Span.xlsx", index=False)

    # Pipeline config that points at this fixture
    cfg = out / "config.fixture.yaml"
    src_cfg = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    shutil.copy(src_cfg, cfg)
    text = cfg.read_text()
    text = text.replace(
        '"./EF_Biomarker/EDF_Files"',
        f'"{(out / "EF_Biomarker" / "EDF_Files").resolve()}"',
    )
    text = text.replace(
        '"./data_root"',
        f'"{(beh_root).resolve()}"',
    )
    text = text.replace(
        '"./results"',
        f'"{(out / "results").resolve()}"',
    )
    text = text.replace(
        '"./figures"',
        f'"{(out / "figures").resolve()}"',
    )
    cfg.write_text(text)

    print(f"\nFixture written to {out}")
    print(f"  Subjects: {args.n_subjects}")
    print(f"  Conditions: {conditions}")
    print(f"  Config: {cfg}")
    print(f"\nRun pipeline:")
    print(f"  python run_all.py --config {cfg}")


if __name__ == "__main__":
    main()
