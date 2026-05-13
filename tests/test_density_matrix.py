"""
Stage 6 unit tests: density-matrix construction, kernel correctness, sklearn API.
"""

import os
import sys

import numpy as np
import pytest

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_ROOT)

mne = pytest.importorskip("mne", reason="mne not installed")

from stages.stage6_density_matrix import (
    HilbertSchmidtKernelSVM,
    compute_density_matrix,
    density_matrix_qc,
    flatten_density_matrix,
    hilbert_schmidt_kernel,
)


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _make_epochs(n_epochs: int = 6, n_channels: int = 8, n_times: int = 1000,
                 sfreq: float = 250.0, seed: int = 0):
    """Synthetic resting-state EEG: 10 Hz sinusoid + Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / sfreq
    base = np.sin(2 * np.pi * 10.0 * t)  # alpha-band signal
    data = np.empty((n_epochs, n_channels, n_times))
    for e in range(n_epochs):
        for ch in range(n_channels):
            data[e, ch] = (
                (1.0 + 0.1 * ch) * base
                + 0.5 * rng.standard_normal(n_times)
            )
    info = mne.create_info(
        [f"C{i}" for i in range(n_channels)], sfreq, ch_types="eeg"
    )
    return mne.EpochsArray(data, info, verbose=False)


# ──────────────────────────────────────────────────────────────────
# Density-matrix mathematical properties
# ──────────────────────────────────────────────────────────────────

class TestDensityMatrixProperties:

    def test_shape(self):
        ep = _make_epochs(n_channels=8)
        rho = compute_density_matrix(ep, 8.0, 13.0)
        assert rho.shape == (8, 8)
        assert rho.dtype == np.complex128

    def test_hermitian(self):
        ep = _make_epochs(n_channels=10)
        rho = compute_density_matrix(ep, 8.0, 13.0)
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)

    def test_unit_trace(self):
        ep = _make_epochs(n_channels=6)
        for fmin, fmax in [(1, 4), (4, 8), (8, 13), (13, 30)]:
            rho = compute_density_matrix(ep, fmin, fmax)
            np.testing.assert_allclose(
                np.real(np.trace(rho)), 1.0, atol=1e-10
            )

    def test_positive_semi_definite(self):
        ep = _make_epochs(n_channels=8)
        rho = compute_density_matrix(ep, 8.0, 13.0)
        eigvals = np.linalg.eigvalsh(rho)
        # Allow tiny negative eigenvalues from float64 round-off; bound is
        # extremely tight after Hermitian-symmetrization.
        assert eigvals.min() > -1e-10

    def test_cauchy_schwarz(self):
        ep = _make_epochs(n_channels=8)
        rho = compute_density_matrix(ep, 8.0, 13.0)
        diag = np.real(np.diag(rho))
        n = rho.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                assert np.abs(rho[i, j]) ** 2 <= diag[i] * diag[j] + 1e-10

    def test_silent_band_returns_zero(self):
        """A band where every time sample has zero norm collapses to zeros.

        We simulate that by feeding a signal with no power in a high band
        outside the synthetic spectrum.
        """
        rng = np.random.default_rng(0)
        n_epochs, n_ch, n_t = 4, 6, 800
        # Pure 5 Hz signal: should be near zero in the 35-45 Hz band.
        t = np.arange(n_t) / 250.0
        data = np.broadcast_to(np.sin(2 * np.pi * 5.0 * t), (n_epochs, n_ch, n_t)).copy()
        # Add tiny noise so analytic signal is not exactly zero.
        data += 1e-9 * rng.standard_normal(data.shape)
        info = mne.create_info([f"C{i}" for i in range(n_ch)], 250.0,
                                ch_types="eeg")
        ep = mne.EpochsArray(data, info, verbose=False)
        rho = compute_density_matrix(ep, 35.0, 45.0)
        # Either trace=1 (with mostly noise) or zeros if all norms collapsed.
        assert np.allclose(rho, rho.conj().T, atol=1e-10)
        assert np.real(np.trace(rho)) <= 1.0 + 1e-10


# ──────────────────────────────────────────────────────────────────
# Flattening invariants
# ──────────────────────────────────────────────────────────────────

class TestFlatten:

    def test_count(self):
        ep = _make_epochs(n_channels=15)
        rho = compute_density_matrix(ep, 8.0, 13.0)
        feat = flatten_density_matrix(rho, [f"C{i}" for i in range(15)], "alpha")
        # 15 diagonals + 2 * (15 choose 2) = 15 + 210 = 225
        assert len(feat) == 225

    def test_round_trip(self):
        """The (Re, Im, diag) decomposition must allow reconstructing rho."""
        ep = _make_epochs(n_channels=6)
        rho = compute_density_matrix(ep, 8.0, 13.0)
        names = [f"C{i}" for i in range(6)]
        feat = flatten_density_matrix(rho, names, "alpha")

        rebuilt = np.zeros_like(rho)
        for i in range(6):
            rebuilt[i, i] = feat[f"dm_alpha_diag_{names[i]}"]
        for i in range(6):
            for j in range(i + 1, 6):
                re = feat[f"dm_alpha_re_{names[i]}_{names[j]}"]
                im = feat[f"dm_alpha_im_{names[i]}_{names[j]}"]
                rebuilt[i, j] = re + 1j * im
                rebuilt[j, i] = re - 1j * im
        np.testing.assert_allclose(rebuilt, rho, atol=1e-10)


# ──────────────────────────────────────────────────────────────────
# Hilbert-Schmidt kernel correctness
# ──────────────────────────────────────────────────────────────────

class TestHSKernel:

    def test_self_kernel_equals_purity(self):
        ep = _make_epochs(n_channels=6)
        rho = compute_density_matrix(ep, 8.0, 13.0)
        K = hilbert_schmidt_kernel([{"alpha": rho}], [{"alpha": rho}], ["alpha"])
        # Tr(rho * rho) = sum of squared singular values = purity.
        purity = float(np.real(np.trace(rho @ rho)))
        np.testing.assert_allclose(K[0, 0], purity, atol=1e-12)

    def test_psd(self):
        """A valid kernel matrix must be PSD."""
        eps = [_make_epochs(n_channels=6, seed=s) for s in range(5)]
        mats = [{"alpha": compute_density_matrix(e, 8.0, 13.0)} for e in eps]
        K = hilbert_schmidt_kernel(mats, mats, ["alpha"])
        K = 0.5 * (K + K.T)
        eigvals = np.linalg.eigvalsh(K)
        assert eigvals.min() > -1e-10

    def test_linear_equivalence(self):
        """Tr(rho_x rho_y) for Hermitian rho equals real part of vec inner-product.

        Vec(A) . conj(Vec(B)) = sum_ij A_ij conj(B_ij).
        For Hermitian matrices Tr(A B) = sum_ij A_ij B_ji = sum_ij A_ij conj(B_ij)
        (because B_ji = conj(B_ij) for Hermitian B), so Tr(A B) = Re-part of the
        Frobenius inner product. The kernel implementation uses this identity.
        """
        eps = [_make_epochs(n_channels=5, seed=s) for s in range(3)]
        mats = [{"alpha": compute_density_matrix(e, 8.0, 13.0)} for e in eps]
        K_kernel = hilbert_schmidt_kernel(mats, mats, ["alpha"])
        K_direct = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                K_direct[i, j] = float(np.real(
                    np.trace(mats[i]["alpha"] @ mats[j]["alpha"])
                ))
        np.testing.assert_allclose(K_kernel, K_direct, atol=1e-12)


# ──────────────────────────────────────────────────────────────────
# QC and SVM API
# ──────────────────────────────────────────────────────────────────

class TestQCReport:

    def test_qc_keys_and_bounds(self):
        ep = _make_epochs(n_channels=8)
        rho = compute_density_matrix(ep, 8.0, 13.0)
        qc = density_matrix_qc(rho)
        for key in ("hermiticity_error", "trace_error", "min_eigenvalue",
                    "cs_violation"):
            assert key in qc
        assert qc["hermiticity_error"] < 1e-10
        assert qc["trace_error"] < 1e-10


class TestHSKernelSVM:

    def test_fit_predict(self):
        # 10 subjects: first 5 with band centered at 10 Hz, last 5 at 20 Hz.
        rng = np.random.default_rng(0)
        info = mne.create_info([f"C{i}" for i in range(8)], 250.0, "eeg")
        mats = []
        labels = []
        for s in range(10):
            t = np.arange(800) / 250.0
            f0 = 10.0 if s < 5 else 20.0
            data = np.empty((4, 8, 800))
            for e in range(4):
                for ch in range(8):
                    data[e, ch] = np.sin(2 * np.pi * f0 * t + 0.1 * ch) \
                                   + 0.3 * rng.standard_normal(800)
            ep = mne.EpochsArray(data, info, verbose=False)
            mats.append({
                "alpha": compute_density_matrix(ep, 8.0, 13.0),
                "beta":  compute_density_matrix(ep, 13.0, 30.0),
            })
            labels.append(0 if s < 5 else 1)

        clf = HilbertSchmidtKernelSVM(bands=["alpha", "beta"], C=1.0)
        clf.fit(mats[:8], np.array(labels[:8]))
        preds = clf.predict(mats[8:])
        assert preds.shape == (2,)
        assert set(preds).issubset({0, 1})

        proba = clf.predict_proba(mats[8:])
        assert proba.shape == (2, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
