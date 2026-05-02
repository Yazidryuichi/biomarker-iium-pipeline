"""
I/O utilities for loading EDF files, behavioral data, and config.

Stage I/O model
---------------
Each pipeline stage owns a folder ``stages/<stage>/`` with its own
``config.yaml``. Stage configs declare ``input.from`` (a predecessor
stage name, ``raw_edf``, or ``behavioral``) and ``output.to`` (a label
that becomes the parent dir under ``results/``).

At runtime ``load_stage_config(stage_name)`` merges the global config
(``configs/config.yaml``: paths, recording, random_state) with the
stage-local params, eagerly creates ``results/<output.to>/<ts>/`` for
output, and resolves ``input_dir`` by either reading globals or finding
the most recent timestamped subdir of the input stage.

Use ``write_stage_notes(output_dir, payload)`` to record an audit trail
(timestamp, git commit, input dirs consumed, outputs produced).
"""

import json
import os
import re
import subprocess
from datetime import datetime

import yaml
import pandas as pd
import numpy as np
from pathlib import Path

STAGE_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path="configs/config.yaml"):
    """Load the global pipeline configuration (paths, recording, random_state)."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_stage_yaml(stage_name):
    """Read the per-stage config file at stages/<stage_name>/config.yaml."""
    path = os.path.join(REPO_ROOT, "stages", stage_name, "config.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Stage config not found: {path}. "
            f"Expected stages/{stage_name}/config.yaml."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_stage_config(stage_name, globals_path="configs/config.yaml",
                      create_output_dir=True):
    """
    Build the merged config dict for a single stage.

    Returns a dict containing:
      paths, recording, random_state          — from globals
      <stage_name>: <params>                   — from stage config
      input_dir, _input_from                   — resolved input location
      output_dir, _output_to                   — newly created output dir
      _stage                                   — stage_name

    Side effect (when ``create_output_dir`` is True): creates
    ``results/<output.to>/<YYYY-MM-DD_HHMMSS>/``. Pass False from
    read-only utilities (e.g. figure generators) that only need
    to inspect input_dir.
    """
    globals_cfg = load_config(globals_path)
    stage_cfg = _load_stage_yaml(stage_name)

    params = stage_cfg.get("params", {}) or {}
    input_decl = stage_cfg.get("input", {}) or {}
    output_decl = stage_cfg.get("output", {}) or {}

    input_from = input_decl.get("from")
    output_to = output_decl.get("to", stage_name)

    merged = {
        "paths": globals_cfg.get("paths", {}),
        "recording": globals_cfg.get("recording", {}),
        "random_state": globals_cfg.get("random_state", 42),
        stage_name: params,
        "_stage": stage_name,
        "_input_from": input_from,
        "_output_to": output_to,
    }

    if input_from == "raw_edf":
        merged["input_dir"] = merged["paths"].get("edf_dir")
    elif input_from == "behavioral":
        merged["input_dir"] = merged["paths"].get("behavioral_dir")
    elif input_from:
        merged["input_dir"] = latest_stage_dir(merged, input_from)
    else:
        merged["input_dir"] = None

    if create_output_dir:
        merged["output_dir"] = make_stage_dir(merged, output_to)
    else:
        merged["output_dir"] = None
    return merged


def get_targets(config):
    """
    Resolve the list of target variables from config.

    Looks up ``analysis.targets`` (list[str]) in the merged stage config.
    Legacy fallbacks: ``analysis.target`` (string), ``ml.targets`` and
    ``ml.target`` (pre-refactor layout). Returns a deduplicated list in
    declared order; falls back to ``["Global_EF"]`` when nothing is set.
    """
    raw = None
    for ns_key in ("analysis", "ml"):
        ns = config.get(ns_key, {}) or {}
        if "targets" in ns:
            raw = ns["targets"]
            break
        if "target" in ns:
            raw = ns["target"]
            break
    if raw is None:
        raw = "Global_EF"
    if isinstance(raw, str):
        raw = [raw]
    seen, out = set(), []
    for t in raw:
        t = str(t).split("#")[0].strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out or ["Global_EF"]


def target_group_col(target):
    """Binary group column name produced by stage engineering for a given target."""
    return f"{target}_group"


# ──────────────────────────────────────────────────────────────────
# Per-stage I/O directory resolution
# ──────────────────────────────────────────────────────────────────

def _results_root(config):
    return config.get("paths", {}).get("results_dir", "./results")


def stage_root_dir(config, stage_name):
    """Return the parent directory holding all timestamped runs for a stage."""
    return os.path.join(_results_root(config), stage_name)


def latest_stage_dir(config, stage_name):
    """
    Return the most recent timestamped subdirectory under
    ``results/<stage_name>/``, or None if no run exists.
    Subdirs are matched against ``YYYY-MM-DD_HHMMSS`` and sorted
    lexicographically (which is also chronological for ISO format).
    """
    root = stage_root_dir(config, stage_name)
    if not os.path.isdir(root):
        return None
    runs = sorted(
        [d for d in os.listdir(root)
         if os.path.isdir(os.path.join(root, d)) and STAGE_TS_RE.match(d)],
        reverse=True,
    )
    return os.path.join(root, runs[0]) if runs else None


def make_stage_dir(config, stage_name):
    """Create ``results/<stage_name>/<YYYY-MM-DD_HHMMSS>/`` and return its path."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(stage_root_dir(config, stage_name), ts)
    os.makedirs(path, exist_ok=True)
    return path


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def write_stage_notes(stage_dir, payload):
    """
    Write ``run_notes.json`` into a stage output directory.
    Adds ``timestamp`` and ``git_commit`` automatically.
    """
    payload = dict(payload)
    payload.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    payload.setdefault("git_commit", _git_commit())
    with open(os.path.join(stage_dir, "run_notes.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def discover_subjects(edf_dir):
    """
    Discover all subjects and their EDF files.
    Returns dict: {subject_id: {condition: filepath}}
    """
    subjects = {}
    for d in sorted(os.listdir(edf_dir)):
        dpath = os.path.join(edf_dir, d)
        if not os.path.isdir(dpath) or not d.startswith("D"):
            continue

        subject_id = d
        subjects[subject_id] = {}

        for f in sorted(os.listdir(dpath)):
            if not f.endswith(".edf"):
                continue

            # Extract condition from filename
            # Pattern: X_M_X_Name_IGS_Eyes_Open.edf or X_M_X_Name_IGS_1_Happy.edf
            # Handle case variants: IGS vs igs, 2_IGS prefix
            fname = f.replace(".edf", "")

            # Normalize: find the condition part after IGS_ or igs_
            match = re.search(r"[Ii][Gg][Ss]_(.+)$", fname)
            if not match:
                continue

            condition_raw = match.group(1)

            # Normalize condition names
            # Handle "2_IGS_" prefix case (subject D0000813 re-recording)
            condition_raw = re.sub(r"^2_IGS_", "", condition_raw)

            # Map to standard condition names
            condition_map = {
                "Eyes_Open": "Eyes_Open",
                "Eyes_Closed": "Eyes_Closed",
                "1_Happy": "1_Happy",
                "2_Calm": "2_Calm",
                "3_Sad": "3_Sad",
                "4_Scare": "4_Scare",
                "5_Wash": "5_Wash",
            }

            condition = condition_map.get(condition_raw, condition_raw)
            subjects[subject_id][condition] = os.path.join(dpath, f)

    return subjects


def load_aufei(filepath):
    """
    Load and score AUFEI-O data.
    Returns DataFrame with subject ID, domain scores, and Global Score.
    """
    df = pd.read_excel(filepath)

    # Identify columns
    wm_cols = [c for c in df.columns if c.startswith("WM")]
    ic_cols = [c for c in df.columns if c.startswith("IC")]
    cf_cols = [c for c in df.columns if c.startswith("CF")]
    p_cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]
    sf_cols = [c for c in df.columns if c.startswith("SF")]

    result = pd.DataFrame()
    result["ID"] = df["ID"]
    result["Sex"] = df["Sex"]
    result["DoB"] = pd.to_datetime(df["DoB"])

    # Domain scores (mean of items)
    result["WM_score"] = df[wm_cols].mean(axis=1) if wm_cols else np.nan
    result["IC_score"] = df[ic_cols].mean(axis=1) if ic_cols else np.nan
    result["CF_score"] = df[cf_cols].mean(axis=1) if cf_cols else np.nan
    result["P_score"] = df[p_cols].mean(axis=1) if p_cols else np.nan
    result["SF_score"] = df[sf_cols].mean(axis=1) if sf_cols else np.nan

    # Global EF: mean of all available subscale mean scores
    subscale_cols = [c for c in ["WM_score", "IC_score", "CF_score", "P_score", "SF_score"]
                     if c in result.columns and result[c].notna().any()]
    result["Global_EF"] = result[subscale_cols].mean(axis=1)

    return result


def _ez_diffusion(mrt_s, vrt_s2, pc, s=0.1):
    """
    EZ-diffusion model (Wagenmakers et al., 2007, Psychonomic Bulletin & Review).
    Returns (v, a, Ter): drift rate, boundary separation, non-decision time.
    Inputs must be in seconds. Returns (nan, nan, nan) on invalid input.
    """
    if not (0 < pc < 1) or vrt_s2 <= 0 or mrt_s <= 0:
        return np.nan, np.nan, np.nan
    pc = np.clip(pc, 0.01, 0.99)
    L = np.log(pc / (1 - pc))
    x = L * (L * pc**2 - L * pc + pc - 0.5) / vrt_s2
    if x <= 0:
        return np.nan, np.nan, np.nan
    v = np.sign(pc - 0.5) * s * x**0.25
    if v == 0:
        return np.nan, np.nan, np.nan
    a = s**2 * L / v
    exp_term = np.exp(-v * a / s**2)
    denom = 1 + exp_term
    ter = mrt_s - (a / (2 * v)) * (1 - exp_term) / denom
    return float(v), float(a), float(ter)


def load_flanker(filepath):
    """Load Flanker Test pilot data and compute EZ-diffusion model parameters."""
    df = pd.read_excel(filepath)

    rt_cols = ["rt_mean", "rt_congruent", "rt_incongruent"]
    for col in rt_cols:
        if col in df.columns and df[col].abs().max() > 1000:
            df[col] = df[col] / 1000.0  # convert ms → seconds

    # EZ-DDM on overall task
    if {"rt_mean", "rt_cv", "acc_overall"}.issubset(df.columns):
        rt_sd = df["rt_cv"] * df["rt_mean"]
        rows = zip(df["rt_mean"], rt_sd**2, df["acc_overall"])
        params = [_ez_diffusion(mrt, vrt, pc) for mrt, vrt, pc in rows]
        df["ddm_v"] = [p[0] for p in params]
        df["ddm_a"] = [p[1] for p in params]
        df["ddm_t"] = [p[2] for p in params]

    # delta_v (congruent − incongruent drift rate): inhibitory control index
    # Use pre-computed column if present, otherwise derive from per-condition RT
    if "ddm_delta_v" not in df.columns:
        if {"rt_congruent", "rt_incongruent", "rt_cv", "acc_overall"}.issubset(df.columns):
            rt_sd = df["rt_cv"] * df["rt_mean"]
            vrt = (rt_sd**2)
            pc = df["acc_overall"]
            v_con = [_ez_diffusion(mrt, vrt_i, pc_i)[0]
                     for mrt, vrt_i, pc_i in zip(df["rt_congruent"], vrt, pc)]
            v_inc = [_ez_diffusion(mrt, vrt_i, pc_i)[0]
                     for mrt, vrt_i, pc_i in zip(df["rt_incongruent"], vrt, pc)]
            df["ddm_delta_v"] = [
                (c - i) if (c is not np.nan and i is not np.nan) else np.nan
                for c, i in zip(v_con, v_inc)
            ]

    return df


def load_digit_span(filepath):
    """Load Digit Span data."""
    return pd.read_excel(filepath)


def get_subject_sex(filename):
    """Extract sex from filename: X_M_X or X_F_X pattern."""
    match = re.search(r"_([MF])_", filename)
    return match.group(1) if match else None
