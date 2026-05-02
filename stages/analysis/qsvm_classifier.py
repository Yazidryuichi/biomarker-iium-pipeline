"""
Quantum Kernel SVM Classifier for EEG Biomarker Classification
===============================================================

Implements a quantum kernel SVM using PennyLane. The classifier
encodes feature vectors into parameterized quantum circuits and
computes the kernel as state fidelity: K(x_i, x_j) = |<phi(x_i)|phi(x_j)>|^2.

This is NOT quantum computing on real hardware. It is a quantum kernel
evaluated on a classical simulator. The value is testing whether a kernel
operating in 2^n-dimensional Hilbert space captures feature interactions
that classical RBF/polynomial kernels miss -- particularly for features
that were themselves extracted using quantum-inspired methods (QEPP,
tensor network entropy, quantum probability).

Design decisions:
  - PCA reduces features to n_qubits dimensions before encoding.
    This is standard practice for quantum kernels (Havlicek et al., 2019).
  - Feature values are scaled to [0, pi] for rotation gate angles.
  - Two circuit variants: ZZ (entangling cross-terms) and RY-only (baseline).
  - sklearn-compatible API: fits into existing Pipeline/cross_validate.

References:
  Havlicek, V., et al. (2019). Supervised learning with quantum-enhanced
    feature spaces. Nature, 567(7747), 209-212.
  Aksoy, G., et al. (2024). Quantum Machine-Based Decision Support System
    for the Detection of Schizophrenia from EEG Records. J. Medical
    Systems, 48(1), 29.

Requires: pip install pennylane
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class QuantumKernelSVM(BaseEstimator, ClassifierMixin):
    """SVM with quantum fidelity kernel computed via PennyLane.

    Parameters
    ----------
    n_qubits : int
        Number of qubits. Input features are PCA-reduced to this dimensionality.
        More qubits = larger Hilbert space (2^n_qubits), but slower kernel.
        Recommended: 4-8 for N < 50 subjects.
    n_layers : int
        Number of encoding layers in the feature map circuit. Each layer
        applies RY/RZ rotations + entangling gates. More layers = more
        expressivity but higher risk of overfitting.
    C : float
        SVM regularization. Lower = more regularization. Important at small N.
    use_entangling : bool
        If True, use ZZ entangling gates between qubits (richer kernel).
        If False, use product-state encoding (no entanglement, baseline).
    random_state : int or None
        Random seed for PCA.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 2,
        C: float = 1.0,
        use_entangling: bool = True,
        random_state: int | None = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.C = C
        self.use_entangling = use_entangling
        self.random_state = random_state

    def _build_kernel_fn(self):
        """Build the PennyLane quantum kernel circuit."""
        import pennylane as qml

        n_q = self.n_qubits
        n_l = self.n_layers
        entangle = self.use_entangling

        dev = qml.device("default.qubit", wires=n_q)

        @qml.qnode(dev)
        def kernel_circuit(x1, x2):
            # Encode x1 into |phi(x1)>
            for layer in range(n_l):
                for i in range(n_q):
                    qml.RY(x1[i], wires=i)
                    qml.RZ(x1[i], wires=i)
                if entangle:
                    for i in range(n_q - 1):
                        qml.CNOT(wires=[i, i + 1])
                    if n_q > 2:
                        qml.CNOT(wires=[n_q - 1, 0])
                    # ZZ cross-terms for richer feature map
                    if layer < n_l - 1:
                        for i in range(n_q - 1):
                            qml.IsingZZ(x1[i] * x1[i + 1], wires=[i, i + 1])

            # Inverse of |phi(x2)> (adjoint encoding)
            for layer in reversed(range(n_l)):
                if entangle and layer < n_l - 1:
                    for i in reversed(range(n_q - 1)):
                        qml.IsingZZ(-x2[i] * x2[i + 1], wires=[i, i + 1])
                if entangle:
                    if n_q > 2:
                        qml.CNOT(wires=[n_q - 1, 0])
                    for i in reversed(range(n_q - 1)):
                        qml.CNOT(wires=[i, i + 1])
                for i in reversed(range(n_q)):
                    qml.RZ(-x2[i], wires=i)
                    qml.RY(-x2[i], wires=i)

            # Probability of measuring |00...0> = fidelity
            return qml.probs(wires=range(n_q))

        return kernel_circuit

    def _compute_kernel_matrix(self, X1, X2):
        """Compute kernel matrix K[i,j] = |<phi(x1_i)|phi(x2_j)>|^2.

        For N=28: 28x28 = 784 circuit evaluations, ~seconds on CPU.
        """
        kernel_fn = self._build_kernel_fn()
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                probs = kernel_fn(X1[i], X2[j])
                K[i, j] = float(probs[0])  # P(|00...0>) = fidelity
        return K

    def _encode_features(self, X, fit=False):
        """PCA reduce + scale to [0, pi] for rotation gate angles."""
        if fit:
            n_components = min(self.n_qubits, X.shape[1], X.shape[0] - 1)
            self._scaler = StandardScaler()
            self._pca = PCA(n_components=n_components, random_state=self.random_state)

            X_scaled = self._scaler.fit_transform(X)
            X_reduced = self._pca.fit_transform(X_scaled)

            # Store range for [0, pi] scaling
            self._feat_min = X_reduced.min(axis=0)
            self._feat_range = np.ptp(X_reduced, axis=0)
            self._feat_range[self._feat_range == 0] = 1.0
        else:
            X_scaled = self._scaler.transform(X)
            X_reduced = self._pca.transform(X_scaled)

        X_normed = (X_reduced - self._feat_min) / self._feat_range * np.pi

        # Pad to n_qubits if PCA gave fewer components
        if X_normed.shape[1] < self.n_qubits:
            pad = np.zeros((X_normed.shape[0], self.n_qubits - X_normed.shape[1]))
            X_normed = np.hstack([X_normed, pad])

        return X_normed

    def fit(self, X, y):
        """Fit QSVM: encode features, compute training kernel, fit SVM."""
        self.classes_ = np.array(sorted(set(y)))
        X_enc = self._encode_features(np.asarray(X, dtype=np.float64), fit=True)
        self._X_train_enc = X_enc

        K_train = self._compute_kernel_matrix(X_enc, X_enc)

        # Symmetrize (numerical precision)
        K_train = (K_train + K_train.T) / 2

        self._svm = SVC(
            kernel="precomputed",
            C=self.C,
            probability=True,
            random_state=self.random_state,
        )
        self._svm.fit(K_train, y)
        return self

    def predict(self, X):
        X_enc = self._encode_features(np.asarray(X, dtype=np.float64))
        K_test = self._compute_kernel_matrix(X_enc, self._X_train_enc)
        return self._svm.predict(K_test)

    def predict_proba(self, X):
        X_enc = self._encode_features(np.asarray(X, dtype=np.float64))
        K_test = self._compute_kernel_matrix(X_enc, self._X_train_enc)
        return self._svm.predict_proba(K_test)

    def __repr__(self):
        ent = "ZZ" if self.use_entangling else "product"
        return (
            f"QuantumKernelSVM(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"C={self.C}, kernel={ent})"
        )
