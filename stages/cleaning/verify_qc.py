"""
Side-by-side QC comparison between two cleaning runs.

Usage:
    python stages/cleaning/verify_qc.py                       # latest two runs
    python stages/cleaning/verify_qc.py OLD_TS NEW_TS         # explicit
    python stages/cleaning/verify_qc.py --old PATH --new PATH # explicit paths

Compares old vs new QC reports across the metrics that matter for the
fix in cleaning.py:
  - reject_threshold_uv variance (old) vs AutoReject thresholds (new)
  - n_ica_excluded distribution per condition (target: median EC >= 1,
    files with 0 IC excluded < 5)
  - pct_epochs_dropped distribution (consistency across subjects)
  - bad channel patterns (Fp1/Fp2/etc frequency)
  - status flips OK <-> LOW_EPOCH_COUNT
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

# Windows cp1252 chokes on the box-drawing/arrow glyphs used below.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def find_latest_two() -> tuple[Path, Path]:
    runs = sorted(p for p in RUNS_DIR.iterdir() if p.is_dir())
    if len(runs) < 2:
        raise RuntimeError(f"Need at least 2 runs in {RUNS_DIR}; found {len(runs)}.")
    return runs[-2], runs[-1]


def load_qc(path: Path) -> list[dict]:
    qc_path = path / "qc.json"
    if not qc_path.exists():
        raise FileNotFoundError(qc_path)
    return json.loads(qc_path.read_text())


def split_by_condition(qc: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for q in qc:
        out.setdefault(q.get("condition", "?"), []).append(q)
    return out


def numeric(values: Iterable, default: float = 0.0) -> list[float]:
    out = []
    for v in values:
        try:
            if v is None:
                continue
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def summarize_numeric(values: Iterable, fmt: str = "{:.1f}") -> str:
    nums = numeric(values)
    if not nums:
        return "—"
    mn, mx = min(nums), max(nums)
    md = stats.median(nums)
    sd = stats.pstdev(nums) if len(nums) > 1 else 0.0
    return (
        f"n={len(nums)} "
        f"median={fmt.format(md)} "
        f"mean={fmt.format(sum(nums)/len(nums))} "
        f"std={fmt.format(sd)} "
        f"range=[{fmt.format(mn)}, {fmt.format(mx)}]"
    )


def reject_threshold_metric(q: dict) -> float | None:
    """Old runs report a single threshold; new runs may use AutoReject."""
    if q.get("autoreject_used"):
        return q.get("autoreject_threshold_uv_mean")
    return q.get("reject_threshold_uv")


def status_map(qc: list[dict]) -> dict[tuple[str, str], str]:
    return {(q["subject"], q["condition"]): q.get("status", "?") for q in qc}


def bad_channel_freq(qc: list[dict]) -> Counter:
    c: Counter = Counter()
    for q in qc:
        for ch in q.get("bad_channels", []) or []:
            c[ch] += 1
    return c


def render_section(title: str, old: str, new: str, marker: str = "") -> str:
    bar = "─" * 76
    return f"\n{bar}\n{title} {marker}\n{bar}\n  OLD : {old}\n  NEW : {new}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ts_old", nargs="?", help="old run timestamp (folder name)")
    parser.add_argument("ts_new", nargs="?", help="new run timestamp (folder name)")
    parser.add_argument("--old", type=Path, help="explicit path to old run dir")
    parser.add_argument("--new", type=Path, help="explicit path to new run dir")
    args = parser.parse_args()

    if args.old and args.new:
        old_dir, new_dir = args.old, args.new
    elif args.ts_old and args.ts_new:
        old_dir = RUNS_DIR / args.ts_old
        new_dir = RUNS_DIR / args.ts_new
    else:
        old_dir, new_dir = find_latest_two()

    print(f"OLD : {old_dir}")
    print(f"NEW : {new_dir}")

    qc_old = load_qc(old_dir)
    qc_new = load_qc(new_dir)

    print(f"\nfiles  OLD={len(qc_old)}  NEW={len(qc_new)}")

    # --- Rejection thresholds (single number per file) ---
    thr_old = [reject_threshold_metric(q) for q in qc_old]
    thr_new = [reject_threshold_metric(q) for q in qc_new]
    print(render_section(
        "Rejection threshold (µV) — variance is the key metric",
        summarize_numeric(thr_old),
        summarize_numeric(thr_new),
        marker="↓std = improvement",
    ))

    # --- IC exclusion distribution per condition ---
    for cond_set in (split_by_condition(qc_old), split_by_condition(qc_new)):
        pass
    cond_old = split_by_condition(qc_old)
    cond_new = split_by_condition(qc_new)
    all_conds = sorted(set(cond_old) | set(cond_new))

    for cond in all_conds:
        ic_o = [q.get("n_ica_excluded", 0) for q in cond_old.get(cond, [])]
        ic_n = [q.get("n_ica_excluded", 0) for q in cond_new.get(cond, [])]
        zero_o = sum(1 for v in ic_o if v == 0)
        zero_n = sum(1 for v in ic_n if v == 0)
        marker = "↑median, ↓zero-count = improvement"
        print(render_section(
            f"n_ica_excluded — condition={cond}",
            f"{summarize_numeric(ic_o, fmt='{:.1f}')}  zero-IC={zero_o}",
            f"{summarize_numeric(ic_n, fmt='{:.1f}')}  zero-IC={zero_n}",
            marker=marker,
        ))

    # --- pct_epochs_dropped distribution ---
    pct_o = [q.get("pct_epochs_dropped", 0) for q in qc_old]
    pct_n = [q.get("pct_epochs_dropped", 0) for q in qc_new]
    print(render_section(
        "pct_epochs_dropped — consistency across subjects",
        summarize_numeric(pct_o, fmt="{:.3f}"),
        summarize_numeric(pct_n, fmt="{:.3f}"),
        marker="↓std = improvement (median may move either way)",
    ))

    # --- Bad channel patterns ---
    bc_o = bad_channel_freq(qc_old)
    bc_n = bad_channel_freq(qc_new)
    keys = sorted(set(bc_o) | set(bc_n))
    print(f"\n{'─'*76}\nBad-channel frequency (count of files flagging each channel)\n{'─'*76}")
    print(f"  {'ch':6}  {'OLD':>5}  {'NEW':>5}  Δ")
    for k in keys:
        d = bc_n[k] - bc_o[k]
        print(f"  {k:6}  {bc_o[k]:>5}  {bc_n[k]:>5}  {d:+d}")

    # --- Status flips ---
    sm_o = status_map(qc_old)
    sm_n = status_map(qc_new)
    flips = []
    only_old, only_new = [], []
    for key in sorted(set(sm_o) | set(sm_n)):
        s_old = sm_o.get(key)
        s_new = sm_n.get(key)
        if s_old is None:
            only_new.append((key, s_new))
        elif s_new is None:
            only_old.append((key, s_old))
        elif s_old != s_new:
            flips.append((key, s_old, s_new))

    print(f"\n{'─'*76}\nStatus flips OLD vs NEW\n{'─'*76}")
    if flips:
        for (sid, cond), so, sn in flips:
            print(f"  {sid} / {cond}: {so} -> {sn}")
    else:
        print("  (none)")
    if only_old:
        print(f"  only in OLD: {len(only_old)} file(s)")
    if only_new:
        print(f"  only in NEW: {len(only_new)} file(s)")

    # --- Targets check ---
    print(f"\n{'─'*76}\nTarget checks (from task spec)\n{'─'*76}")
    ec_new = cond_new.get("Eyes_Closed", [])
    ic_n_ec = [q.get("n_ica_excluded", 0) for q in ec_new]
    median_ec_new = stats.median(ic_n_ec) if ic_n_ec else 0
    zero_new_all = sum(1 for q in qc_new if q.get("n_ica_excluded", 0) == 0)
    zero_old_all = sum(1 for q in qc_old if q.get("n_ica_excluded", 0) == 0)
    std_thr_old = stats.pstdev(numeric(thr_old)) if numeric(thr_old) else 0
    std_thr_new = stats.pstdev(numeric(thr_new)) if numeric(thr_new) else 0
    std_pct_old = stats.pstdev(numeric(pct_o)) if numeric(pct_o) else 0
    std_pct_new = stats.pstdev(numeric(pct_n)) if numeric(pct_n) else 0

    def tick(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    pass_ec = median_ec_new >= 1
    pass_zero = zero_new_all < 5
    pass_thr_var = std_thr_new < std_thr_old
    pass_pct_var = std_pct_new <= std_pct_old * 1.10  # within 10% counts as no regression

    print(f"  [{tick(pass_ec)}] median n_ica_excluded (EC) >= 1   "
          f"(got {median_ec_new})")
    print(f"  [{tick(pass_zero)}] zero-IC files (all) < 5            "
          f"(old={zero_old_all} → new={zero_new_all})")
    print(f"  [{tick(pass_thr_var)}] threshold variance ↓             "
          f"(std: {std_thr_old:.1f} → {std_thr_new:.1f} µV)")
    print(f"  [{tick(pass_pct_var)}] pct_dropped variance not worse  "
          f"(std: {std_pct_old:.3f} → {std_pct_new:.3f})")

    regressions = []
    if not pass_ec: regressions.append("median EC n_ica_excluded")
    if not pass_zero: regressions.append("zero-IC count")
    if not pass_thr_var: regressions.append("threshold variance")
    if not pass_pct_var: regressions.append("pct_dropped variance")
    if flips:
        ok_to_low = [f for f in flips if f[1] == "OK" and f[2] == "LOW_EPOCH_COUNT"]
        if ok_to_low:
            regressions.append(f"{len(ok_to_low)} OK→LOW_EPOCH flips")

    print(f"\n{'═'*76}\nVERDICT: ", end="")
    if regressions:
        print("REGRESSION in: " + "; ".join(regressions))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
