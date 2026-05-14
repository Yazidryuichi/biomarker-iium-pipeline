# ============================================================
# EDF to CSV Batch Converter
# Input : folder berisi file .edf (rekursif, per-participant)
# Output: .csv di folder mirror + annotations terpisah (jika ada)
# ============================================================

import mne
import pandas as pd
from pathlib import Path

# --- KONFIGURASI ---
INPUT_ROOT  = Path(r"C:\Users\Dandy\Documents\Project\EEG_Preprocessing\split_ID")
OUTPUT_ROOT = Path(r"C:\Users\Dandy\Documents\Project\EEG_Preprocessing\output_csv")
TIME_FORMAT = "ms"        # "ms" atau "datetime"
PICKS       = None        # None = semua channel, atau list e.g. ["Fz","Cz","Pz"]
DROP_LABEL  = True        # drop channel "LABEL" quirk WinEEG
OVERWRITE   = False       # True = timpa CSV yang sudah ada

# --- MAIN ---
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
edf_files = sorted(INPUT_ROOT.rglob("*.edf"))
print(f"Ditemukan {len(edf_files)} file EDF\n")

log = []
for i, edf_path in enumerate(edf_files, 1):
    rel = edf_path.relative_to(INPUT_ROOT)
    out_csv = OUTPUT_ROOT / rel.with_suffix(".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not OVERWRITE:
        print(f"[{i}/{len(edf_files)}] SKIP (exists): {rel}")
        continue

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

        # Drop LABEL channel quirk
        if DROP_LABEL and "LABEL" in raw.ch_names:
            raw.drop_channels(["LABEL"])

        # Bersihkan suffix -A1/-A2
        rename = {ch: ch.split("-")[0] for ch in raw.ch_names if "-" in ch}
        if rename:
            raw.rename_channels(rename)

        # Export sinyal
        df = raw.to_data_frame(picks=PICKS, time_format=TIME_FORMAT)
        df.to_csv(out_csv, index=False)

        # Export annotations (jika ada)
        if len(raw.annotations) > 0:
            ann_path = out_csv.with_name(out_csv.stem + "_annotations.csv")
            raw.annotations.to_data_frame().to_csv(ann_path, index=False)

        size_mb = out_csv.stat().st_size / 1e6
        print(f"[{i}/{len(edf_files)}] OK  : {rel}  ({size_mb:.1f} MB, "
              f"{len(raw.ch_names)} ch, {raw.n_times} samples)")
        log.append({"file": str(rel), "status": "OK",
                    "n_channels": len(raw.ch_names),
                    "n_samples": raw.n_times, "size_mb": round(size_mb, 2)})

    except Exception as e:
        print(f"[{i}/{len(edf_files)}] ERR : {rel}  -> {e}")
        log.append({"file": str(rel), "status": f"ERROR: {e}"})

# Save summary log
pd.DataFrame(log).to_csv(OUTPUT_ROOT / "_conversion_log.csv", index=False)
print(f"\nSelesai. Log: {OUTPUT_ROOT / '_conversion_log.csv'}")