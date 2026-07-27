"""
Stage 2: Validation
===================
Reliability + target-quality checks on behavioural measures BEFORE we spend
compute on EEG features. Trial-level when available; summary-level only when
the source workbook genuinely lacks trial data.

Outputs (output/<ts>/):
  aufei_subscale_reliability.csv   Cronbach alpha per subscale + interpretation
  aufei_item_total.csv             corrected item-total r + alpha-if-deleted
  aufei_global_reliability.csv     Cronbach alpha + omega for Global_EF composite
  aufei_global_item_total.csv      corrected item-total r for the global composite
  flanker_reliability.csv          split-half Spearman-Brown per metric
  digit_span_reliability.csv       FW vs BW split-half (parallel-halves assumption)
  preprocessing_pass_summary.json  N retained after Stage 1 min_epochs floor
  reliability_summary.json         consolidated headline
  run_notes.json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

STAGE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = STAGE_DIR.parent
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")


def load_config():
    with open(STAGE_DIR / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def resolve(path):
    p = Path(path)
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
    """Find the most recent timestamped subdir under <stage>/output/."""
    root = resolve(root)
    if not root.is_dir():
        return None
    runs = sorted([d for d in root.iterdir()
                   if d.is_dir() and TS_RE.match(d.name)], reverse=True)
    return runs[0] if runs else None


# ──────────────────────────────────────────────────────────────────
# AUFEI: item-level reliability
# ──────────────────────────────────────────────────────────────────

def cronbach_alpha(items_df):
    """Classical Cronbach's alpha. items_df: subjects x items, items numeric."""
    X = items_df.apply(pd.to_numeric, errors="coerce").dropna()
    k = X.shape[1]
    n = X.shape[0]
    if k < 2 or n < 3:
        return float("nan"), n
    item_var = X.var(axis=0, ddof=1).sum()
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return float("nan"), n
    a = (k / (k - 1.0)) * (1.0 - item_var / total_var)
    return float(a), n


def mcdonald_omega(items_df):
    """One-factor congeneric omega via factor_analyzer; nan if unavailable."""
    try:
        from factor_analyzer import FactorAnalyzer
    except Exception:
        return float("nan")
    X = items_df.apply(pd.to_numeric, errors="coerce").dropna()
    if X.shape[0] < 5 or X.shape[1] < 3:
        return float("nan")
    try:
        fa = FactorAnalyzer(n_factors=1, rotation=None, method="minres")
        fa.fit(X.values)
        loadings = fa.loadings_.ravel()
        uniq = fa.get_uniquenesses()
        num = (loadings.sum()) ** 2
        den = num + uniq.sum()
        return float(num / den) if den > 0 else float("nan")
    except Exception:
        return float("nan")


def item_total_diagnostics(items_df):
    """Corrected item-total r + alpha if item deleted, per item."""
    X = items_df.apply(pd.to_numeric, errors="coerce").dropna()
    rows = []
    cols = list(X.columns)
    base_alpha, _ = cronbach_alpha(X)
    for col in cols:
        rest = X[[c for c in cols if c != col]].sum(axis=1)
        r = float(np.corrcoef(X[col], rest)[0, 1]) if X[col].std() > 0 else float("nan")
        a_if_deleted, _ = cronbach_alpha(X[[c for c in cols if c != col]])
        rows.append({
            "item": col,
            "corrected_item_total_r": round(r, 3) if np.isfinite(r) else None,
            "alpha_if_deleted": round(a_if_deleted, 3) if np.isfinite(a_if_deleted) else None,
            "alpha_baseline": round(base_alpha, 3) if np.isfinite(base_alpha) else None,
        })
    return rows


def aufei_reliability(beh_dir, fname, subscales, alpha_min, alpha_pref):
    path = beh_dir / fname
    if not path.exists():
        return None, None, {"error": f"missing {path}"}
    df = pd.read_excel(path)

    subscale_rows = []
    item_rows = []
    for name, items in subscales.items():
        present = [c for c in items if c in df.columns]
        if len(present) < 2:
            subscale_rows.append({
                "subscale": name, "k_items": len(present), "n_obs": 0,
                "alpha": None, "omega": None, "interpretation": "insufficient items",
            })
            continue
        sub = df[present]
        a, n = cronbach_alpha(sub)
        w = mcdonald_omega(sub)
        if not np.isfinite(a):
            interp = "undefined"
        elif a >= alpha_pref:
            interp = "good"
        elif a >= alpha_min:
            interp = "marginal"
        else:
            interp = "inadequate"
        subscale_rows.append({
            "subscale": name, "k_items": len(present), "n_obs": int(n),
            "alpha": round(a, 3) if np.isfinite(a) else None,
            "omega": round(w, 3) if np.isfinite(w) else None,
            "interpretation": interp,
        })
        for r in item_total_diagnostics(sub):
            r["subscale"] = name
            item_rows.append(r)

    return (pd.DataFrame(subscale_rows),
            pd.DataFrame(item_rows),
            {"alpha_min_acceptable": alpha_min, "alpha_preferred": alpha_pref})


def aufei_global_reliability(beh_dir, fname, subscales, drop_items,
                             alpha_min, alpha_pref):
    """Cronbach alpha + omega for the single item-level Global_EF composite.

    Mirrors feature_engineering.load_aufei: the item set is the union of all
    subscale items minus `drop_items` (NOT a mean of subscale means). This is
    the reliability of the confirmatory-path composite; per CLAUDE.md report the
    global alpha/omega, never per-subscale, on the confirmatory path. Does not
    replace the per-subscale audit above — both are emitted.
    """
    path = beh_dir / fname
    if not path.exists():
        return None, None, {"error": f"missing {path}"}
    df = pd.read_excel(path)
    drop = set(drop_items or [])
    all_items = [it for items in subscales.values() for it in items]
    present = [c for c in all_items if c in df.columns and c not in drop]
    if len(present) < 2:
        return None, None, {"error": "insufficient global items present"}

    sub = df[present]
    a, n = cronbach_alpha(sub)
    w = mcdonald_omega(sub)
    if not np.isfinite(a):
        interp = "undefined"
    elif a >= alpha_pref:
        interp = "good"
    elif a >= alpha_min:
        interp = "marginal"
    else:
        interp = "inadequate"
    global_row = pd.DataFrame([{
        "composite": "Global_EF",
        "k_items": len(present),
        "n_obs": int(n),
        "alpha": round(a, 3) if np.isfinite(a) else None,
        "omega": round(w, 3) if np.isfinite(w) else None,
        "interpretation": interp,
        "note": "single item-level composite over all retained AUFEI-O items; "
                "high alpha is partly item-count driven, not strong "
                "unidimensionality (see CLAUDE.md)",
    }])
    item_rows = item_total_diagnostics(sub)
    for r in item_rows:
        r["composite"] = "Global_EF"
    return (global_row, pd.DataFrame(item_rows),
            {"items_used": present, "n_items": len(present),
             "n_dropped": len(set(all_items) - set(present))})


# ──────────────────────────────────────────────────────────────────
# Flanker: trial-level split-half reliability
# ──────────────────────────────────────────────────────────────────

def ez_diffusion(mrt, vrt, pc, s=0.1):
    """Wagenmakers et al. 2007 EZ-DDM. Returns (v, a, Ter); nans if undefined."""
    if not (0 < pc < 1) or vrt <= 0 or mrt <= 0:
        return float("nan"), float("nan"), float("nan")
    pc = float(np.clip(pc, 0.01, 0.99))
    L = np.log(pc / (1 - pc))
    x = L * (L * pc**2 - L * pc + pc - 0.5) / vrt
    if x <= 0:
        return float("nan"), float("nan"), float("nan")
    v = np.sign(pc - 0.5) * s * x**0.25
    if v == 0:
        return float("nan"), float("nan"), float("nan")
    a = s**2 * L / v
    exp_t = np.exp(-v * a / s**2)
    ter = mrt - (a / (2 * v)) * (1 - exp_t) / (1 + exp_t)
    return float(v), float(a), float(ter)


def per_half_metrics(trials):
    """trials: DataFrame with columns acc, rt, condition. Returns dict of metrics.

    Per-condition EZ-DDM (ddm_v_congruent, ddm_v_incongruent) and the
    derived ddm_delta_v are computed here so their split-half reliability
    can be reported alongside the components. EZ-DDM is undefined at
    acc in {0, 1}; halves with degenerate accuracy yield NaN, which is
    then dropped from the cross-subject correlation.
    """
    valid = trials.dropna(subset=["acc", "rt"])
    if "outlier" in valid.columns:
        valid = valid[~valid["outlier"].astype(bool)]
    if len(valid) == 0:
        return {}
    out = {}

    # Overall accuracy + RT
    out["acc_overall"] = float(valid["acc"].mean())
    out["rt_mean"] = float(valid["rt"].mean())
    rt_sd = float(valid["rt"].std(ddof=1)) if len(valid) > 1 else float("nan")
    out["rt_cv"] = (rt_sd / out["rt_mean"]
                    if np.isfinite(rt_sd) and out["rt_mean"] > 0
                    else float("nan"))

    # Per-condition accuracy + RT
    inc = valid[valid["condition"].astype(str).str.lower() == "incongruent"]
    con = valid[valid["condition"].astype(str).str.lower() == "congruent"]
    out["acc_congruent"]   = float(con["acc"].mean()) if len(con) else float("nan")
    out["acc_incongruent"] = float(inc["acc"].mean()) if len(inc) else float("nan")
    out["rt_incongruent"]  = float(inc["rt"].mean()) if len(inc) else float("nan")
    out["rt_congruent"]    = float(con["rt"].mean()) if len(con) else float("nan")
    out["flanker_effect"]  = (out["rt_incongruent"] - out["rt_congruent"]
                              if np.isfinite(out["rt_incongruent"]) and
                                 np.isfinite(out["rt_congruent"]) else float("nan"))

    # Overall EZ-DDM
    vrt_all = float(valid["rt"].var(ddof=1)) if len(valid) > 1 else float("nan")
    v, a, t = ez_diffusion(out["rt_mean"], vrt_all, out["acc_overall"])
    out["ddm_v"] = v; out["ddm_a"] = a; out["ddm_t"] = t

    # Per-condition EZ-DDM
    for cond_name, cond_df in (("congruent", con), ("incongruent", inc)):
        if len(cond_df) > 1:
            cpc = float(cond_df["acc"].mean())
            cmrt = float(cond_df["rt"].mean())
            cvrt = float(cond_df["rt"].var(ddof=1))
            cv, ca, ct = ez_diffusion(cmrt, cvrt, cpc)
        else:
            cv = float("nan")
        out[f"ddm_v_{cond_name}"] = cv

    # ddm_delta_v = v_con - v_inc (signed magnitude of slowing under conflict).
    # Difference score — reliability typically much lower than its components.
    vc, vi = out.get("ddm_v_congruent"), out.get("ddm_v_incongruent")
    out["ddm_delta_v"] = (vc - vi
                          if (vc is not None and vi is not None
                              and np.isfinite(vc) and np.isfinite(vi))
                          else float("nan"))
    return out


def split_half_reliability(per_subject_halves, metrics):
    """per_subject_halves: dict[sid] = (half1_metrics, half2_metrics)."""
    rows = []
    for m in metrics:
        x1, x2 = [], []
        for sid, (h1, h2) in per_subject_halves.items():
            v1 = h1.get(m); v2 = h2.get(m)
            if v1 is None or v2 is None:
                continue
            if not (np.isfinite(v1) and np.isfinite(v2)):
                continue
            x1.append(v1); x2.append(v2)
        n = len(x1)
        if n < 4:
            rows.append({"metric": m, "n": n, "r_halves": None,
                         "spearman_brown": None,
                         "note": "n<4 after dropping undefined halves"})
            continue
        r = float(np.corrcoef(x1, x2)[0, 1])
        sb = (2 * r) / (1 + r) if (1 + r) != 0 else float("nan")
        rows.append({
            "metric": m, "n": n,
            "r_halves": round(r, 3),
            "spearman_brown": round(sb, 3) if np.isfinite(sb) else None,
            "note": "",
        })
    return pd.DataFrame(rows)


def flanker_reliability(beh_dir, fname, split_mode, min_per_half, metrics, seed):
    path = beh_dir / fname
    if not path.exists():
        return None, {"error": f"missing {path}"}
    xl = pd.ExcelFile(path)
    if "Trials" not in xl.sheet_names:
        return None, {"error": "no Trials sheet in workbook"}
    trials = pd.read_excel(path, sheet_name="Trials")
    needed = {"id", "trial", "condition", "acc", "rt"}
    if not needed.issubset(trials.columns):
        return None, {"error": f"Trials sheet missing columns: {needed - set(trials.columns)}"}

    halves = {}
    rng = np.random.default_rng(seed)
    for sid, g in trials.groupby("id"):
        g = g.sort_values("trial").reset_index(drop=True)
        if split_mode == "random":
            idx = rng.permutation(len(g))
            mid = len(idx) // 2
            h1 = g.iloc[idx[:mid]]; h2 = g.iloc[idx[mid:2 * mid]]
        else:  # odd_even
            h1 = g[g["trial"] % 2 == 0]
            h2 = g[g["trial"] % 2 == 1]
        if len(h1) < min_per_half or len(h2) < min_per_half:
            continue
        halves[sid] = (per_half_metrics(h1), per_half_metrics(h2))

    rel_df = split_half_reliability(halves, metrics)
    meta = {
        "n_subjects_total": int(trials["id"].nunique()),
        "n_subjects_used": len(halves),
        "split_mode": split_mode,
        "min_trials_per_half": min_per_half,
        "trials_total_rows": int(len(trials)),
    }
    return rel_df, meta


# ──────────────────────────────────────────────────────────────────
# Digit Span: FW vs BW as approximate parallel halves
# ──────────────────────────────────────────────────────────────────

def digit_span_reliability(beh_dir, fname):
    path = beh_dir / fname
    if not path.exists():
        return None, {"error": f"missing {path}"}
    df = pd.read_excel(path)
    if not {"FW_Raw", "BW_Raw"}.issubset(df.columns):
        return None, {"error": "FW_Raw or BW_Raw missing"}
    sub = df[["FW_Raw", "BW_Raw"]].dropna()
    if len(sub) < 4:
        return None, {"error": f"n={len(sub)} <4"}
    r = float(np.corrcoef(sub["FW_Raw"], sub["BW_Raw"])[0, 1])
    sb = (2 * r) / (1 + r) if (1 + r) != 0 else float("nan")
    row = pd.DataFrame([{
        "measure": "Total_Raw (FW+BW)",
        "n": int(len(sub)),
        "r_halves": round(r, 3),
        "spearman_brown": round(sb, 3) if np.isfinite(sb) else None,
        "note": "FW and BW are NOT strictly parallel; FW = short-term retention, "
                "BW adds manipulation. Report as approximate test-reliability proxy, "
                "not definitive.",
    }])
    return row, {"n_used": int(len(sub))}


# ──────────────────────────────────────────────────────────────────
# Preprocessing pass-rate summary
# ──────────────────────────────────────────────────────────────────

def preprocessing_pass_summary(preproc_root):
    latest = latest_output(preproc_root)
    if latest is None:
        return {"error": f"no preprocessing run under {preproc_root}"}
    qc_path = latest / "qc.json"
    if not qc_path.exists():
        return {"error": f"qc.json missing in {latest}"}
    with open(qc_path, "r") as f:
        qc = json.load(f)
    by_status = {}
    for q in qc:
        by_status[q.get("status", "UNKNOWN")] = by_status.get(q.get("status", "UNKNOWN"), 0) + 1
    subjects = set()
    conds_per_subject = {}
    for q in qc:
        if q.get("status") == "OK":
            sid = q.get("subject"); subjects.add(sid)
            conds_per_subject.setdefault(sid, set()).add(q.get("condition"))
    n_both_conditions = sum(1 for c in conds_per_subject.values() if len(c) >= 2)
    return {
        "preprocessing_run_dir": str(latest),
        "n_records_total": len(qc),
        "n_by_status": by_status,
        "n_subjects_with_at_least_one_ok_condition": len(subjects),
        "n_subjects_with_both_eo_and_ec": n_both_conditions,
    }


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    p = cfg["params"]
    beh_dir = resolve(cfg["paths"]["behavioral_dir"])
    out_dir = make_output_dir()
    print(f"Validation output: {out_dir}")
    print(f"Behavioral dir: {beh_dir}")

    headline = {}

    # AUFEI
    print("\n--- AUFEI item-level reliability ---")
    aufei_sub, aufei_items, aufei_meta = aufei_reliability(
        beh_dir, p["aufei_filename"], p["aufei_subscales"],
        float(p["alpha_min_acceptable"]), float(p["alpha_preferred"]))
    if aufei_sub is not None:
        aufei_sub.to_csv(out_dir / "aufei_subscale_reliability.csv", index=False)
        aufei_items.to_csv(out_dir / "aufei_item_total.csv", index=False)
        print(aufei_sub.to_string(index=False))
        headline["aufei"] = {
            "subscales": aufei_sub.to_dict(orient="records"),
            **aufei_meta,
        }
    else:
        print(f"  ERROR: {aufei_meta}")
        headline["aufei"] = {"error": str(aufei_meta)}

    # AUFEI — Global_EF composite reliability (confirmatory-path composite)
    print("\n--- AUFEI Global_EF reliability (all retained items) ---")
    gef_df, gef_items, gef_meta = aufei_global_reliability(
        beh_dir, p["aufei_filename"], p["aufei_subscales"],
        p.get("aufei_drop_items", []),
        float(p["alpha_min_acceptable"]), float(p["alpha_preferred"]))
    if gef_df is not None:
        gef_df.to_csv(out_dir / "aufei_global_reliability.csv", index=False)
        gef_items.to_csv(out_dir / "aufei_global_item_total.csv", index=False)
        print(gef_df.to_string(index=False))
        print(f"  {gef_meta}")
        headline["aufei_global"] = {
            "composite": gef_df.to_dict(orient="records")[0],
            **gef_meta,
        }
    else:
        print(f"  ERROR: {gef_meta}")
        headline["aufei_global"] = {"error": str(gef_meta)}

    # Flanker — trial-level split-half (the main fix)
    print("\n--- Flanker split-half reliability (trial-level) ---")
    flank_df, flank_meta = flanker_reliability(
        beh_dir, p["flanker_filename"], p["flanker_split"],
        int(p["flanker_min_trials_per_half"]), p["flanker_metrics"],
        cfg["random_state"])
    if flank_df is not None:
        flank_df.to_csv(out_dir / "flanker_reliability.csv", index=False)
        print(flank_df.to_string(index=False))
        print(f"  {flank_meta}")
        headline["flanker"] = {
            "metrics": flank_df.to_dict(orient="records"),
            **flank_meta,
        }
    else:
        print(f"  ERROR: {flank_meta}")
        headline["flanker"] = {"error": str(flank_meta)}

    # Digit Span — FW vs BW (approximate)
    print("\n--- Digit Span FW-vs-BW (approximate parallel halves) ---")
    ds_df, ds_meta = digit_span_reliability(beh_dir, p["digit_span_filename"])
    if ds_df is not None:
        ds_df.to_csv(out_dir / "digit_span_reliability.csv", index=False)
        print(ds_df.to_string(index=False))
        headline["digit_span"] = {
            "measure": ds_df.to_dict(orient="records")[0],
            **ds_meta,
        }
    else:
        headline["digit_span"] = {"error": str(ds_meta)}
        print(f"  ERROR: {ds_meta}")

    # Preprocessing pass summary
    print("\n--- Preprocessing pass summary ---")
    pp_summary = preprocessing_pass_summary(cfg["paths"]["preprocessing_root"])
    with open(out_dir / "preprocessing_pass_summary.json", "w") as f:
        json.dump(pp_summary, f, indent=2, default=str)
    print(json.dumps(pp_summary, indent=2))
    headline["preprocessing_pass"] = pp_summary

    # Consolidated summary
    with open(out_dir / "reliability_summary.json", "w") as f:
        json.dump(headline, f, indent=2, default=str)

    notes = {
        "stage": "validation",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "behavioral_dir": str(beh_dir),
        "preprocessing_consumed": pp_summary.get("preprocessing_run_dir"),
        "outputs": ["aufei_subscale_reliability.csv", "aufei_item_total.csv",
                    "aufei_global_reliability.csv", "aufei_global_item_total.csv",
                    "flanker_reliability.csv", "digit_span_reliability.csv",
                    "preprocessing_pass_summary.json", "reliability_summary.json"],
    }
    with open(out_dir / "run_notes.json", "w") as f:
        json.dump(notes, f, indent=2, default=str)

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
