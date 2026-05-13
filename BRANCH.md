# Branch: `quantum-exploration`

This branch is **Option B** of the Phase 3 hybrid framing decision: keep the
quantum-cognition interpretive layer for future strengthening at N = 100,
separated from the `main` branch (which is **Option A**: non-linear feature
transforms with no quantum-physics analogy).

## Why this branch exists

The senior-scientist critique on the pre-Phase-1 portfolio flagged two distinct
issues with the original "quantum-inspired" framing:

1. **Methodological** (addressed on `main` in Phase 1) — the apples-to-oranges
   feature × model comparison + per-fold CIs too tight at small N. Both are
   now rewritten with the fair 2×2 comparison + subject-level LOSO + paired
   DeLong + subject-bootstrap CIs.

2. **Framing** — the "quantum-inspired" branding implies physical-quantum
   content that the features (von Neumann entropy of an EEG covariance matrix,
   density-matrix Hilbert-Schmidt similarity) don't have without a separate
   argument. The user's hybrid call: **main = drop the framing, present as
   non-linear feature transforms**; **this branch = keep the framing and
   build out the supporting argument for N = 100**.

## What's different on this branch

Currently nothing — the branch is forked from `main` at commit `0f8b426`
(Phase 2 commit). The differentiation is **forward-looking**:

- This branch will accumulate methods-level work that interprets the
  density-matrix features through the quantum-cognition / quantum-probability
  lens (Busemeyer & Bruza 2012, Khrennikov & Yamada 2025, Alotaibi et al.
  2026).
- A future deliverable on this branch is a methods proof that DM features
  cannot be matched by an equivalent classical non-linear transform
  (e.g., kernel PCA on covariance features). Until that proof lands, the
  "quantum cognition reveals something a non-linear classical method couldn't"
  claim is conjectural.
- The branch is **reserved for future strengthening at N = 100**. Don't
  expect to ship from it until the replication arm produces signal.

## What stays accessible on `main`

The quantum-cognition feature-extraction code (`stages/exploratory_quantum.py`,
`stages/qsvm_classifier.py`) is still **present on `main`** and runnable via:

```bash
python run_all.py --exploratory-quantum
```

But it is no longer part of the **default numbered pipeline flow** on `main`
(it was Stage 5 pre-Phase-1; that slot now belongs to the fair-comparison
analysis). The next iteration of Phase 3 may decide to physically remove
these files from `main` once the branch differentiation is settled.

## Reading order for a reviewer arriving here

1. `main` branch `README.md` — what the pipeline does today.
2. `main` branch `METHODS.md` §4 (density-matrix feature extraction) —
   the mathematics, presented without quantum-cognition framing.
3. (This branch only) future `METHODS_QUANTUM_INTERPRETATION.md` — the
   quantum-cognition argument for why the density-matrix features carry
   information classical non-linear transforms don't.

## Status

- Branch created: 2026-05-13 (Phase 3 of `UPGRADE_PROPOSAL_v1.md`).
- Active work: paused; awaiting N = 100 cohort + replication on ds004284.
- Maintenance: rebase periodically against `main` to keep methods updates in
  sync (the framing differs, but the feature extraction code shouldn't drift).
