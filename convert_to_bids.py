"""
Convert raw EDF data to BIDS format
=====================================
Uses mne-bids to organize raw EEG data into Brain Imaging Data
Structure (BIDS) format for standardized data sharing.

Usage:
    python convert_to_bids.py
    python convert_to_bids.py --output bids_dataset

Requirements:
    pip install mne-bids

References:
    - BIDS-EEG: Pernet et al. (2019) Scientific Data
    - OpenNeuro ds005305 as structural reference
"""

import argparse
import json
import os
import sys

import mne
import numpy as np
import pandas as pd

PIPELINE_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_ROOT)

from utils.io import load_config, discover_subjects


def convert_to_bids(config, output_dir="bids_dataset"):
    """
    Convert raw EDF files to BIDS-compliant structure.

    Creates:
        bids_dataset/
            dataset_description.json
            participants.tsv
            participants.json
            sub-XXX/
                eeg/
                    sub-XXX_task-restingstate_eeg.edf
                    sub-XXX_task-restingstate_eeg.json
                    sub-XXX_task-restingstate_channels.tsv
                    sub-XXX_task-restingstate_events.tsv
    """
    try:
        from mne_bids import write_raw_bids, BIDSPath, make_dataset_description
    except ImportError:
        print("ERROR: mne-bids not installed. Run: pip install mne-bids")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Create dataset description
    make_dataset_description(
        path=output_dir,
        name="QEEG Biomarker for Executive Function in Indonesian Children",
        authors=[
            "Dandy Aulya",
            "S.Y. Dewi",
            "Yazid Ryuichi Habiburahman",
        ],
        data_type="eeg",
        how_to_acknowledge="Please cite: [publication pending]",
        ethics_approvals=["RS Soeharto Heerdjan Ethics Committee"],
        funding=["Self-funded"],
        references_and_links=["https://github.com/Yazidryuichi/biomarker-iium-pipeline"],
        overwrite=True,
    )

    # Load behavioral data for participants.tsv
    beh_dir = config["paths"]["behavioral_dir"]
    aufei_path = os.path.join(beh_dir, "AUFEI-O", "AUFEI-O_Cleaned.xlsx")
    try:
        from utils.io import load_aufei
        aufei = load_aufei(aufei_path)
        aufei_dict = {row["ID"]: row for _, row in aufei.iterrows()}
    except Exception:
        aufei_dict = {}

    # Discover subjects
    subjects = discover_subjects(config["paths"]["edf_dir"])
    print(f"Found {len(subjects)} subjects to convert")

    participants_data = []
    task_map = {
        "Eyes_Open": "restEO",
        "Eyes_Closed": "restEC",
    }

    for sub_id in sorted(subjects.keys()):
        sub_label = sub_id.replace("D", "")  # BIDS: numeric only

        for condition, filepath in subjects[sub_id].items():
            task = task_map.get(condition)
            if task is None:
                continue  # Skip emotional conditions for BIDS

            try:
                raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)

                # Set channel types
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

                bids_path = BIDSPath(
                    subject=sub_label,
                    task=task,
                    datatype="eeg",
                    root=output_dir,
                )

                write_raw_bids(
                    raw, bids_path,
                    format="EDF",
                    overwrite=True,
                    verbose=False,
                )
                print(f"  {sub_id}/{condition} -> {bids_path}")

            except Exception as e:
                print(f"  ERROR {sub_id}/{condition}: {e}")

        # Collect participant info
        info = aufei_dict.get(sub_id, {})
        participants_data.append({
            "participant_id": f"sub-{sub_label}",
            "age": info.get("age_years", "n/a") if isinstance(info, dict)
                   else getattr(info, "age_years", "n/a"),
            "sex": info.get("Sex", "n/a") if isinstance(info, dict)
                   else getattr(info, "Sex", "n/a"),
        })

    # Write participants.tsv
    participants_df = pd.DataFrame(participants_data)
    participants_df.to_csv(
        os.path.join(output_dir, "participants.tsv"),
        sep="\t", index=False,
    )

    # Write participants.json (field descriptions)
    participants_json = {
        "participant_id": {
            "Description": "Unique subject identifier"
        },
        "age": {
            "Description": "Age at time of assessment in years",
            "Units": "years"
        },
        "sex": {
            "Description": "Biological sex",
            "Levels": {
                "M": "Male",
                "F": "Female"
            }
        },
    }
    with open(os.path.join(output_dir, "participants.json"), "w") as f:
        json.dump(participants_json, f, indent=2)

    print(f"\nBIDS conversion complete: {output_dir}/")
    print(f"Subjects: {len(participants_data)}")
    print(f"Validate with: bids-validator {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert raw EDF to BIDS format")
    parser.add_argument("--output", type=str, default="bids_dataset",
                        help="Output directory for BIDS dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    os.chdir(PIPELINE_ROOT)
    config = load_config(args.config)
    convert_to_bids(config, args.output)
