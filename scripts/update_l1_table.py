#!/usr/bin/env python3
"""Auto-fill the L1 sensitivity TBD table in README from stage5_fair_comparison.json.

Usage:
    # After running Stage 5 with --include-l1-sensitivity:
    python -m stages.stage5_fair_comparison \
        --results-dir results \
        --out-json results/stage5_fair_comparison.json \
        --include-l1-sensitivity

    # Then update the README table:
    python scripts/update_l1_table.py

Reads `results/stage5_fair_comparison.json`, locates the L1 sensitivity cells
(`classical+svm_l1`, `classical+lr_l1`, `classical+lr_elasticnet`), and
rewrites the README "Sensitivity analysis" subsection's TBD table with real
numbers. Also fills in a reading guide based on whether each L1 cell lifts
above 0.5 (chance).

Idempotent: rerunning produces the same output. Safe to re-invoke after
re-running Stage 5 with different settings.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


L1_CELLS = ("classical+svm_l1", "classical+lr_l1", "classical+lr_elasticnet")

CELL_LABEL = {
    "classical+svm_l1": "classical+svm_l1",
    "classical+lr_l1": "classical+lr_l1",
    "classical+lr_elasticnet": "classical+lr_elasticnet",
}


def _format_auc_with_ci(cell: dict) -> str:
    auc = cell.get("loso_auc_mean")
    lo = cell.get("loso_auc_ci_lo")
    hi = cell.get("loso_auc_ci_hi")
    if auc is None:
        return "N/A"
    if lo is None or hi is None:
        return f"{auc:.3f}"
    return f"{auc:.3f} [{lo:.3f}, {hi:.3f}]"


def _read_delong_p_vs_dm_svm(pairwise: list, target_cell: str) -> str:
    """Find the DeLong test result where one cell is density_matrix+svm_linear
    and the other is `target_cell`. Returns formatted p or 'N/A'."""
    for d in pairwise:
        cells = {d["cell_1"], d["cell_2"]}
        if cells == {"density_matrix+svm_linear", target_cell}:
            p = d.get("delong_p_two_sided")
            if p is None:
                return "N/A"
            sig = " ✓" if p < 0.05 else ""
            return f"{p:.3f}{sig}"
    return "N/A (cell missing)"


def _reading_for(name: str, auc: float | None) -> str:
    if auc is None:
        return "(not run)"
    if auc < 0.5:
        verdict = "still below chance"
    elif auc < 0.6:
        verdict = "near chance"
    else:
        verdict = "above chance"
    short = {
        "classical+svm_l1": f"L1-SVC: {verdict}",
        "classical+lr_l1": f"LR-L1: {verdict}",
        "classical+lr_elasticnet": f"LR-elasticnet: {verdict}",
    }
    return short.get(name, verdict)


def build_new_table(stage5_json: dict) -> str:
    """Construct the replacement Markdown table from the Stage 5 JSON."""
    cells_by_name = {c["cell"]: c for c in stage5_json.get("cells", [])}
    pairwise = stage5_json.get("pairwise_delong", [])

    # Baseline row (always present in canonical 2x2)
    baseline = cells_by_name.get("classical+svm_linear", {})
    baseline_auc_ci = _format_auc_with_ci(baseline)
    baseline_p = _read_delong_p_vs_dm_svm(pairwise, "classical+svm_linear")

    rows = [
        "| Cell | LOSO AUC [95% CI] | DeLong p vs DM+svm_linear | Reading |",
        "|---|---|---|---|",
        f"| classical+svm_linear (baseline) | {baseline_auc_ci} | {baseline_p} | "
        f"{_reading_for('classical+svm_linear', baseline.get('loso_auc_mean'))} (overfitting at N=28) |",
    ]
    for cell_name in L1_CELLS:
        cell = cells_by_name.get(cell_name, {})
        auc_ci = _format_auc_with_ci(cell)
        p = _read_delong_p_vs_dm_svm(pairwise, cell_name)
        reading = _reading_for(cell_name, cell.get("loso_auc_mean"))
        if not cell:
            reading = "(not in JSON; rerun with --include-l1-sensitivity)"
        rows.append(f"| {CELL_LABEL[cell_name]} | {auc_ci} | {p} | {reading} |")
    return "\n".join(rows)


def patch_readme(readme_path: Path, new_table: str) -> bool:
    """Find the L1 sensitivity table block in README and replace it.

    Looks for the block starting with the markdown heading
    `| Cell | LOSO AUC [95% CI] | DeLong p vs DM+svm_linear | Reading |`
    and replaces through the last `classical+lr_elasticnet | ... |` row.

    Returns True if the README was modified, False if no block was found
    or the block already matched (no-op).
    """
    text = readme_path.read_text()

    # Match the table block: header line through the lr_elasticnet row.
    pattern = re.compile(
        r"\| Cell \| LOSO AUC \[95% CI\] \| DeLong p vs DM\+svm_linear \| Reading \|\n"
        r"\|[^\n]+\|\n"  # the alignment row
        r"(?:\|[^\n]+\|\n)+",
        re.MULTILINE,
    )

    match = pattern.search(text)
    if not match:
        return False

    if match.group(0).rstrip("\n") == new_table:
        return False  # no-op

    new_text = text[: match.start()] + new_table + "\n" + text[match.end():]
    readme_path.write_text(new_text)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-json", type=Path,
        default=Path("results/stage5_fair_comparison.json"),
        help="Path to the Stage 5 results JSON written by "
             "stages.stage5_fair_comparison --include-l1-sensitivity",
    )
    parser.add_argument(
        "--readme", type=Path, default=Path("README.md"),
        help="Path to the README to patch (default: ./README.md)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the proposed new table; don't modify the README",
    )
    args = parser.parse_args()

    if not args.results_json.exists():
        print(f"ERROR: {args.results_json} does not exist. Run Stage 5 first:",
              file=sys.stderr)
        print("  python -m stages.stage5_fair_comparison \\", file=sys.stderr)
        print(f"      --out-json {args.results_json} \\", file=sys.stderr)
        print("      --include-l1-sensitivity", file=sys.stderr)
        return 2

    if not args.readme.exists():
        print(f"ERROR: {args.readme} does not exist", file=sys.stderr)
        return 2

    stage5 = json.loads(args.results_json.read_text())
    cell_names = {c["cell"] for c in stage5.get("cells", [])}
    missing = [c for c in L1_CELLS if c not in cell_names]
    if missing:
        print(f"WARNING: L1 sensitivity cells missing from JSON: {missing}",
              file=sys.stderr)
        print("Did you forget --include-l1-sensitivity?", file=sys.stderr)
        # Still proceed — the table will show "(not in JSON)" for missing rows

    new_table = build_new_table(stage5)

    if args.dry_run:
        print("--- proposed new table (dry-run) ---")
        print(new_table)
        return 0

    changed = patch_readme(args.readme, new_table)
    if changed:
        print(f"Updated {args.readme} L1 sensitivity table.")
        return 0
    print(f"No change to {args.readme} (table block missing, or already up to date).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
