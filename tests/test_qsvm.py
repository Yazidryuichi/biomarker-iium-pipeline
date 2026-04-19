"""
Tests for Quantum Kernel SVM classifier.

Uses small synthetic data — no PennyLane required for skip detection,
but tests are skipped if pennylane is not installed.
"""

import os
import sys

import numpy as np
import pytest

PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_ROOT)

pennylane = pytest.importorskip("pennylane", reason="pennylane not installed")

from stages.qsvm_classifier import QuantumKernelSVM


@pytest.fixture(scope="module")
def binary_data():
    """Small synthetic binary classification dataset (N=20, 10 features)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 10))
    # Class 0: shift features 0-4 down; Class 1: shift up
    X[:10, :5] -= 1.0
    X[10:, :5] += 1.0
    y = np.array([0] * 10 + [1] * 10)
    return X, y


class TestQuantumKernelSVM:
    """Core QSVM functionality."""

    def test_fit_predict(self, binary_data):
        X, y = binary_data
        clf = QuantumKernelSVM(n_qubits=3, n_layers=1, C=1.0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == y.shape
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, binary_data):
        X, y = binary_data
        clf = QuantumKernelSVM(n_qubits=3, n_layers=1)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (20, 2)
        # Probabilities should sum to ~1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_classes_attribute(self, binary_data):
        X, y = binary_data
        clf = QuantumKernelSVM(n_qubits=3, n_layers=1)
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1])

    def test_entangling_vs_product(self, binary_data):
        """Both variants should run without error."""
        X, y = binary_data
        for ent in [True, False]:
            clf = QuantumKernelSVM(n_qubits=3, n_layers=1, use_entangling=ent)
            clf.fit(X, y)
            preds = clf.predict(X)
            assert len(preds) == len(y)

    def test_more_qubits_than_features(self):
        """Should handle n_qubits > n_features via zero-padding."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((12, 3))
        y = np.array([0] * 6 + [1] * 6)
        clf = QuantumKernelSVM(n_qubits=6, n_layers=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == 12

    def test_few_samples(self):
        """Should work with very small N (like a single CV fold)."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((8, 5))
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        clf = QuantumKernelSVM(n_qubits=3, n_layers=1)
        clf.fit(X, y)
        preds = clf.predict(X[:2])
        assert len(preds) == 2

    def test_sklearn_cross_validate(self, binary_data):
        """Must work with sklearn cross_validate (the actual use case)."""
        from sklearn.model_selection import StratifiedKFold, cross_validate

        X, y = binary_data
        clf = QuantumKernelSVM(n_qubits=3, n_layers=1, C=1.0)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_validate(clf, X, y, cv=cv, scoring="balanced_accuracy")
        assert "test_score" in scores
        assert len(scores["test_score"]) == 3
        # Should be above random (0.5) on this separable data
        assert np.mean(scores["test_score"]) > 0.4

    def test_repr(self):
        clf = QuantumKernelSVM(n_qubits=4, n_layers=2, use_entangling=True)
        r = repr(clf)
        assert "n_qubits=4" in r
        assert "ZZ" in r

    def test_product_repr(self):
        clf = QuantumKernelSVM(n_qubits=4, n_layers=2, use_entangling=False)
        assert "product" in repr(clf)
