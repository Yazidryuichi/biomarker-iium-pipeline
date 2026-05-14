"""
Builds notebooks/desc_validity.ipynb from cell definitions below.

This builder exists because the notebook is mostly mechanical assembly of
print() calls, descriptive tables, and correlation matrices — easier to
maintain as a Python source file than as raw .ipynb JSON. Run from repo
root: `python scripts/_build_desc_validity_notebook.py`.
"""
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "notebooks" / "desc_validity.ipynb"


def md(text: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in textwrap.dedent(text).strip("\n").splitlines()],
    }


def code(src: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in textwrap.dedent(src).strip("\n").splitlines()],
    }


CELLS = [
    md("""
        # Behavioral measures — descriptives & validity

        Pilot draft companion to `scripts/reliability_analysis.py`. Run cell-by-cell;
        each cell ends with a DataFrame so Jupyter renders it directly.

        Scope:

        1. **Descriptives** for AUFEI-O (per item + per subscale + Global),
           Digit Span, Flanker (RT, accuracy, EZ-DDM parameters), with normality
           checks (skew, kurtosis, Shapiro–Wilk).
        2. **Convergent validity** — AUFEI subscale inter-correlations.
        3. **Cross-measure validity** — pre-specified hypothesised pairings
           (AUFEI WM ↔ Digit Span, AUFEI IC ↔ Flanker conflict cost),
           BH-FDR corrected.
        4. **Discriminant** — full behavioural correlation matrix for context.

        Sample N ≈ 26–28 depending on cleaning floor. Treat everything as
        preliminary; report point estimates with CIs and Spearman alongside
        Pearson because of the small N + non-normality.
    """),
    code("""
        # ── Setup ─────────────────────────────────────────────────────────
        import os, sys
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from scipy import stats
        from statsmodels.stats.multitest import multipletests

        ROOT = Path(os.getcwd()).resolve()
        if ROOT.name == "notebooks":
            ROOT = ROOT.parent          # running from inside notebooks/
        sys.path.insert(0, str(ROOT))
        from utils.io import load_aufei, load_flanker, load_digit_span  # noqa: E402

        pd.set_option("display.float_format", lambda x: f"{x:.3f}")
        pd.set_option("display.max_columns", 60)
        pd.set_option("display.width", 200)

        BEH = ROOT / "data" / "Behavioral"
        print(f"Repo root: {ROOT}")
        print(f"Behavioral dir: {BEH}  exists={BEH.exists()}")
    """),
    md("""
        ## 1. Load raw data

        Three workbooks. `load_aufei` already computes subscale + Global means
        from item-level scores; we keep both the scored frame (`aufei`) and the
        raw item frame (`aufei_raw`) because item-level stats need the latter.
    """),
    code("""
        aufei_raw = pd.read_excel(BEH / "AUFEI-O_Cleaned.xlsx")
        aufei = load_aufei(BEH / "AUFEI-O_Cleaned.xlsx")
        flanker = load_flanker(BEH / "Flanker_Test_Pilot.xlsx")
        digit = load_digit_span(BEH / "Digit_Span.xlsx")

        print(f"AUFEI  raw  : {aufei_raw.shape}  ({aufei_raw['ID'].nunique()} unique IDs)")
        print(f"AUFEI scored: {aufei.shape}")
        print(f"Flanker     : {flanker.shape}")
        print(f"Digit Span  : {digit.shape}")
        aufei.head()
    """),
    md("""
        ## 2. Demographics

        Sex × age distribution. Pilot is balanced 13/13. Age range 6.8–12.8 yr.
    """),
    code("""
        # Pull assessment_date from the engineering config so age matches
        # whatever the pipeline last used. Falls back to today if unavailable.
        import yaml
        eng_cfg_path = ROOT / "stages" / "engineering" / "config.yaml"
        with open(eng_cfg_path, "r", encoding="utf-8") as _f:
            _eng = yaml.safe_load(_f)
        ref_date = pd.Timestamp(_eng.get("params", {}).get("assessment_date") or pd.Timestamp.today())
        print(f"Reference date for age: {ref_date.date()}")

        demo = aufei.copy()
        demo["age_years"] = (ref_date - pd.to_datetime(demo["DoB"])).dt.days / 365.25

        demo_summary = (
            demo.groupby("Sex")
                .agg(N=("ID", "nunique"),
                     age_mean=("age_years", "mean"),
                     age_sd=("age_years", "std"),
                     age_min=("age_years", "min"),
                     age_max=("age_years", "max"))
                .round(2)
        )
        demo_summary.loc["Total"] = [
            demo["ID"].nunique(),
            demo["age_years"].mean(),
            demo["age_years"].std(),
            demo["age_years"].min(),
            demo["age_years"].max(),
        ]
        demo_summary.round(2)
    """),
    md("""
        ## 3. AUFEI-O — item-level descriptives

        5 items per subscale, 4-point Likert (observed range 1–4). Look for:

        - **Constant items** (sd ≈ 0): contribute no information, will be
          flagged in reliability output too. (IC3 in pilot — all subjects = 4.)
        - **Strong floor/ceiling**: |skew| > 1.0 is worth noting.
    """),
    code("""
        SUBSCALES = {
            "WM": ["WM1","WM2","WM3","WM4","WM5"],
            "IC": ["IC1","IC2","IC3","IC4","IC5"],
            "CF": ["CF1","CF2","CF3","CF4","CF5"],
            "P":  ["P1","P2","P3","P4","P5"],
            "SF": ["SF1","SF2","SF3","SF4","SF5"],
        }

        rows = []
        for sub, items in SUBSCALES.items():
            for it in items:
                s = aufei_raw[it].dropna()
                rows.append({
                    "subscale": sub,
                    "item": it,
                    "N": int(s.size),
                    "mean": s.mean(),
                    "sd": s.std(ddof=1),
                    "median": s.median(),
                    "min": s.min(),
                    "max": s.max(),
                    "skew": stats.skew(s, bias=False) if s.std(ddof=1) > 0 else np.nan,
                    "kurtosis": stats.kurtosis(s, bias=False) if s.std(ddof=1) > 0 else np.nan,
                    "pct_at_max": (s == s.max()).mean() * 100,
                })
        item_desc = pd.DataFrame(rows).set_index(["subscale","item"]).round(3)
        item_desc
    """),
    md("""
        ## 4. AUFEI-O — subscale + Global descriptives

        Subscale scores are item means (range 1–4). `pct_at_ceiling` = share of
        subjects whose subscale mean ≥ 3.5 (rough indicator of right-skew on a
        parent-report scale).
    """),
    code("""
        sub_cols = ["WM_score","IC_score","CF_score","P_score","SF_score","Global_EF"]
        rows = []
        for c in sub_cols:
            s = aufei[c].dropna()
            shapiro_W, shapiro_p = stats.shapiro(s) if s.size >= 3 else (np.nan, np.nan)
            rows.append({
                "measure": c,
                "N": int(s.size),
                "mean": s.mean(),
                "sd": s.std(ddof=1),
                "median": s.median(),
                "iqr": s.quantile(0.75) - s.quantile(0.25),
                "min": s.min(),
                "max": s.max(),
                "skew": stats.skew(s, bias=False),
                "kurtosis": stats.kurtosis(s, bias=False),
                "shapiro_W": shapiro_W,
                "shapiro_p": shapiro_p,
                "pct_at_ceiling": (s >= 3.5).mean() * 100,
            })
        aufei_desc = pd.DataFrame(rows).set_index("measure").round(3)
        print("Shapiro p<.05 → reject normality; expect this on small samples.")
        aufei_desc
    """),
    md("""
        ## 5. AUFEI subscale × Sex

        Pilot is balanced, so this is exploratory only. Welch's t (unequal var)
        + Mann–Whitney U for non-parametric back-up. Cohen's d uses pooled SD.
    """),
    code("""
        merged_demo = aufei.merge(demo[["ID","age_years"]], on="ID", how="left")
        rows = []
        for c in sub_cols:
            m = merged_demo.loc[merged_demo["Sex"] == "Laki-laki", c].dropna()
            f = merged_demo.loc[merged_demo["Sex"] == "Perempuan", c].dropna()
            if min(len(m), len(f)) < 3:
                continue
            t, p_t = stats.ttest_ind(m, f, equal_var=False)
            u, p_u = stats.mannwhitneyu(m, f, alternative="two-sided")
            pooled_sd = np.sqrt(((m.var(ddof=1) * (len(m)-1)) + (f.var(ddof=1) * (len(f)-1))) /
                                 (len(m) + len(f) - 2))
            d = (m.mean() - f.mean()) / pooled_sd if pooled_sd > 0 else np.nan
            rows.append({
                "measure": c,
                "n_M": len(m), "n_F": len(f),
                "mean_M": m.mean(), "mean_F": f.mean(),
                "sd_M": m.std(ddof=1), "sd_F": f.std(ddof=1),
                "cohen_d": d,
                "welch_t": t, "welch_p": p_t,
                "mwu_U": u,   "mwu_p": p_u,
            })
        pd.DataFrame(rows).set_index("measure").round(3)
    """),
    md("""
        ## 6. AUFEI subscale × Age

        Pearson + Spearman r with age in years. AUFEI is an age-normed
        rating — in a healthy pilot we'd typically expect modest positive
        correlations or none, not strong slopes.
    """),
    code("""
        rows = []
        for c in sub_cols:
            d = merged_demo[[c, "age_years"]].dropna()
            if len(d) < 4:
                continue
            r_p, p_p = stats.pearsonr(d[c], d["age_years"])
            r_s, p_s = stats.spearmanr(d[c], d["age_years"])
            rows.append({
                "measure": c, "N": len(d),
                "pearson_r": r_p,  "pearson_p": p_p,
                "spearman_r": r_s, "spearman_p": p_s,
            })
        pd.DataFrame(rows).set_index("measure").round(3)
    """),
    md("""
        ## 7. Digit Span — descriptives

        Six columns from the workbook. `Total_*` are sums of FW and BW.
        Distributions tend to be near-normal-ish at this age range.
    """),
    code("""
        ds_cols = ["FW_Span","FW_Raw","BW_Span","BW_Raw","Total_Span","Total_Raw"]
        rows = []
        for c in ds_cols:
            if c not in digit.columns:
                continue
            s = digit[c].dropna()
            sw_W, sw_p = stats.shapiro(s) if s.size >= 3 else (np.nan, np.nan)
            rows.append({
                "measure": c, "N": int(s.size),
                "mean": s.mean(), "sd": s.std(ddof=1),
                "median": s.median(),
                "iqr": s.quantile(0.75) - s.quantile(0.25),
                "min": s.min(), "max": s.max(),
                "skew": stats.skew(s, bias=False),
                "kurtosis": stats.kurtosis(s, bias=False),
                "shapiro_W": sw_W, "shapiro_p": sw_p,
            })
        pd.DataFrame(rows).set_index("measure").round(3)
    """),
    md("""
        ## 8. Flanker — descriptives

        Three families of columns reported separately:

        - **Accuracy**: `acc_overall`, `acc_incongruent` (ceiling-prone)
        - **RT** (sec): `rt_mean`, `rt_congruent`, `rt_incongruent`, `flanker_effect`, `rt_cv`
        - **EZ-DDM**: `ddm_v_*`, `ddm_a_*`, `ddm_t0_*`, `ddm_delta_v`
        - Trial counts: `n_trials`, `n_outliers`
    """),
    code("""
        def _desc(col):
            s = flanker[col].dropna()
            if s.size < 3:
                return None
            sw_W, sw_p = stats.shapiro(s)
            return {
                "measure": col, "N": int(s.size),
                "mean": s.mean(), "sd": s.std(ddof=1),
                "median": s.median(),
                "iqr": s.quantile(0.75) - s.quantile(0.25),
                "min": s.min(), "max": s.max(),
                "skew": stats.skew(s, bias=False),
                "kurtosis": stats.kurtosis(s, bias=False),
                "shapiro_W": sw_W, "shapiro_p": sw_p,
            }

        groups = {
            "Accuracy": ["acc_overall","acc_incongruent"],
            "RT (s)":   ["rt_mean","rt_congruent","rt_incongruent",
                         "flanker_effect","rt_cv","rt_iqr","block_effect_rt"],
            "IES":      ["ies_congruent","ies_incongruent"],
            "EZ-DDM":   ["ddm_v_congruent","ddm_v_incongruent","ddm_delta_v",
                         "ddm_a_congruent","ddm_a_incongruent",
                         "ddm_t0_congruent","ddm_t0_incongruent"],
            "Trials":   ["n_trials","n_outliers","pct_outliers"],
        }

        frames = []
        for g, cols in groups.items():
            rows = [_desc(c) for c in cols if c in flanker.columns]
            rows = [r for r in rows if r is not None]
            if not rows:
                continue
            df_g = pd.DataFrame(rows)
            df_g.insert(0, "family", g)
            frames.append(df_g)
        flanker_desc = pd.concat(frames, ignore_index=True).set_index(["family","measure"]).round(4)
        flanker_desc
    """),
    md("""
        ## 9. Convergent validity — AUFEI subscale inter-correlations

        Pearson + Spearman in one frame. With N≈28 and ordinal scoring,
        Spearman is the safer headline. We expect moderate positive
        correlations because subscales share a common EF construct, but very
        high (> .8) would suggest the subscales aren't discriminating
        sub-domains.
    """),
    code("""
        def _pair_corr(df, x, y):
            d = df[[x, y]].dropna()
            if len(d) < 4:
                return None
            r_p, p_p = stats.pearsonr(d[x], d[y])
            r_s, p_s = stats.spearmanr(d[x], d[y])
            return {
                "x": x, "y": y, "N": len(d),
                "pearson_r": r_p,  "pearson_p": p_p,
                "spearman_r": r_s, "spearman_p": p_s,
            }

        subs = ["WM_score","IC_score","CF_score","P_score","SF_score","Global_EF"]
        rows = []
        for i, a in enumerate(subs):
            for b in subs[i+1:]:
                r = _pair_corr(aufei, a, b)
                if r:
                    rows.append(r)
        aufei_internal = pd.DataFrame(rows).round(3)
        # FDR-BH over Spearman p-values (within this family of tests).
        if not aufei_internal.empty:
            _, p_fdr, _, _ = multipletests(aufei_internal["spearman_p"], method="fdr_bh")
            aufei_internal["spearman_p_fdr"] = np.round(p_fdr, 3)
        aufei_internal
    """),
    md("""
        ## 10. Hypothesised cross-measure validity

        Pre-specified pairings (theory-driven, NOT data-driven):

        | AUFEI domain | External marker | Direction |
        |---|---|---|
        | WM_score | Digit Span Total_Raw, BW_Raw, FW_Raw | + (parent rating of WM should align with span) |
        | IC_score | Flanker `flanker_effect` (neg sign), `rt_incongruent`, `ddm_delta_v` (+) | rating of impulse control ↔ less conflict cost / better drift |
        | CF_score | Flanker `block_effect_rt` (neg) | flexible switching reduces fatigue/block drag |
        | Global_EF | Digit Span Total_Raw, Flanker `ddm_delta_v`, `acc_overall` | composite EF ↔ both task families |

        BH-FDR applied across the full hypothesis family below.
    """),
    code("""
        beh = (aufei.merge(digit, on="ID", how="inner")
                    .merge(flanker, on="ID", how="inner"))
        print(f"Merged behavioural N = {beh.shape[0]}  (intersection of 3 files)")

        HYPOTHESES = [
            # (aufei col, external col, expected sign)
            ("WM_score",     "Total_Raw",       +1),
            ("WM_score",     "BW_Raw",          +1),
            ("WM_score",     "FW_Raw",          +1),
            ("IC_score",     "flanker_effect",  -1),
            ("IC_score",     "rt_incongruent",  -1),
            ("IC_score",     "ddm_delta_v",     +1),
            ("CF_score",     "block_effect_rt", -1),
            ("Global_EF",    "Total_Raw",       +1),
            ("Global_EF",    "ddm_delta_v",     +1),
            ("Global_EF",    "acc_overall",     +1),
        ]

        rows = []
        for x, y, sign in HYPOTHESES:
            if x not in beh.columns or y not in beh.columns:
                rows.append({"x": x, "y": y, "expected_sign": sign,
                             "note": "column missing"})
                continue
            d = beh[[x, y]].dropna()
            if len(d) < 4:
                rows.append({"x": x, "y": y, "expected_sign": sign, "N": len(d),
                             "note": "n<4"})
                continue
            r_p, p_p = stats.pearsonr(d[x], d[y])
            r_s, p_s = stats.spearmanr(d[x], d[y])
            rows.append({
                "x": x, "y": y, "expected_sign": sign, "N": len(d),
                "pearson_r": r_p,  "pearson_p": p_p,
                "spearman_r": r_s, "spearman_p": p_s,
                "sign_match": "yes" if (np.sign(r_s) == sign) else "no",
            })
        hypotheses_df = pd.DataFrame(rows)
        # FDR within the family of tests that produced a p-value.
        mask = hypotheses_df["spearman_p"].notna()
        if mask.any():
            _, p_fdr, _, _ = multipletests(hypotheses_df.loc[mask, "spearman_p"], method="fdr_bh")
            hypotheses_df.loc[mask, "spearman_p_fdr"] = np.round(p_fdr, 3)
        hypotheses_df.round(3)
    """),
    md("""
        ## 11. Full cross-measure correlation matrix (Spearman)

        Discriminant check: for completeness, all behavioural composites
        vs. each other. Bold pattern to look for in interpretation:

        - AUFEI block (top-left) should cluster (covered in §9).
        - Digit Span FW/BW/Total should cluster.
        - Flanker RT family should cluster, with `flanker_effect` showing
          weaker / inverse signs vs. accuracy & DDM drift.
        - Cross-block correlations should generally be smaller in magnitude
          than within-block, supporting domain discriminant validity.
    """),
    code("""
        target_cols = [
            "WM_score","IC_score","CF_score","P_score","SF_score","Global_EF",
            "FW_Raw","BW_Raw","Total_Raw",
            "acc_overall","rt_mean","flanker_effect","rt_cv",
            "ddm_v_congruent","ddm_v_incongruent","ddm_delta_v",
        ]
        target_cols = [c for c in target_cols if c in beh.columns]
        corr_s = beh[target_cols].corr(method="spearman").round(2)
        corr_s
    """),
    md("""
        ## 12. Pairwise p-values for the matrix above (BH-FDR within the off-diagonal)

        Useful for paper supplementary. Significance after FDR is conservative
        with this many pairs and N≈28; treat any survivor as a candidate
        rather than confirmation.
    """),
    code("""
        rows = []
        cols = target_cols
        for i, a in enumerate(cols):
            for b in cols[i+1:]:
                d = beh[[a, b]].dropna()
                if len(d) < 4:
                    continue
                r, p = stats.spearmanr(d[a], d[b])
                rows.append({"x": a, "y": b, "N": len(d),
                             "spearman_r": r, "spearman_p": p})
        long_corr = pd.DataFrame(rows)
        if not long_corr.empty:
            _, p_fdr, _, _ = multipletests(long_corr["spearman_p"], method="fdr_bh")
            long_corr["spearman_p_fdr"] = p_fdr
        long_corr["sig_05_fdr"] = long_corr["spearman_p_fdr"] < 0.05
        long_corr.sort_values("spearman_p").round(3).head(30)
    """),
    md("""
        ## 13. Quick takeaways for the paper draft

        Re-execute after re-running cleaning/engineering so N matches the
        final EEG cohort. The numbers above are based on the **raw**
        behavioural files (N=28), not the post-cleaning intersection (N≈26).
        For Table 1, prefer the intersection sample — re-run §3, §4, §7, §8
        after merging `aufei` against `stages/engineering/runs/<latest>/full_dataset.csv`.

        Suggested manuscript wording:

        > Internal consistency of AUFEI-O subscales ranged from poor
        > (Working Memory α = −0.13, Inhibitory Control α = 0.25) to
        > moderate (Planning α = 0.68, Social Functioning α = 0.64,
        > Cognitive Flexibility α = 0.59), with the 25-item Global score
        > showing acceptable reliability (α = 0.81, ω = 0.82). Convergent
        > validity against task-based measures was strongest for the Global
        > composite; subscale-level relationships were directionally
        > consistent with theory but did not survive FDR correction in the
        > pilot (N = …). These results are preliminary and motivate item
        > review (reverse-coding of WM4/WM5; near-constant IC3) prior to
        > full-sample collection.
    """),
]

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"Wrote {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
