# Copyright 2026 QHARM
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2, QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms import VQD
from qiskit_algorithms.algorithm_job import AlgorithmJob
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit_algorithms.optimizers import COBYLA, SLSQP
from qiskit_algorithms.state_fidelities import (
    BaseStateFidelity,
    StateFidelityResult,
)


class _ExactStatevectorFidelity(BaseStateFidelity):
    def create_fidelity_circuit(self, circuit_1, circuit_2):
        return circuit_1.copy()

    def _run(
        self,
        circuits_1,
        circuits_2,
        values_1=None,
        values_2=None,
        *,
        shots=None,
    ):
        if isinstance(circuits_1, QuantumCircuit):
            circuits_1 = [circuits_1]
        if isinstance(circuits_2, QuantumCircuit):
            circuits_2 = [circuits_2]
        values_1 = self._preprocess_values(circuits_1, values_1)
        values_2 = self._preprocess_values(circuits_2, values_2)
        # _preprocess_values returns a single empty row when values is None
        if len(values_1) == 1 and len(circuits_1) > 1:
            values_1 = list(values_1) * len(circuits_1)
        if len(values_2) == 1 and len(circuits_2) > 1:
            values_2 = list(values_2) * len(circuits_2)

        def _call():
            fidelities = []
            for circuit_1, circuit_2, val_1, val_2 in zip(
                circuits_1, circuits_2, values_1, values_2
            ):
                bound_1 = (
                    circuit_1.assign_parameters(val_1) if val_1 else circuit_1
                )
                bound_2 = (
                    circuit_2.assign_parameters(val_2) if val_2 else circuit_2
                )
                overlap = Statevector(bound_1).inner(Statevector(bound_2))
                fidelities.append(float(np.abs(overlap) ** 2))
            return StateFidelityResult(
                fidelities=fidelities,
                raw_fidelities=fidelities,
                metadata=[{} for _ in fidelities],
                shots=0,
            )

        return AlgorithmJob(_call)


def _statevector_to_real_eigenvector(circuit, parameters, n):
    # Turn the VQD circuit into a real unit vector of length n.
    bound = circuit.assign_parameters(parameters)
    vec = np.asarray(Statevector(bound).data[:n], dtype=complex)
    idx = int(np.argmax(np.abs(vec)))
    peak = vec[idx]
    if np.abs(peak) > 0:
        vec = vec * np.conj(peak) / np.abs(peak)
    vec = np.real(vec)
    if vec[idx] < 0:
        vec = -vec
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _matrix_to_observable(flat_matrix, n):
    # Reshape into an n x n NumPy array and pad to the next power of two.
    mat = np.array(flat_matrix, dtype=float).reshape((n, n))
    next_pow2 = 1 << (n - 1).bit_length()
    if next_pow2 != n:
        padded_mat = np.zeros((next_pow2, next_pow2))
        padded_mat[:n, :n] = mat
        # Pad unused diagonal with Gershgorin bound so fake eigenvalues stay outside.
        gershgorin = float(np.max(np.sum(np.abs(mat), axis=1)))
        penalty = max(1.0, gershgorin)
        for i in range(n, next_pow2):
            padded_mat[i, i] = penalty
        mat = padded_mat
    return SparsePauliOp.from_operator(mat), next_pow2


def _make_estimator(use_noise):
    if not use_noise:
        return StatevectorEstimator(), None
    try:
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager,
        )
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
        from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    except ImportError as exc:
        raise ImportError(
            "Noisy solvers require qiskit-aer and qiskit-ibm-runtime. "
            "Install with: pip install qiskit-aer qiskit-ibm-runtime"
        ) from exc

    fake_backend = FakeManilaV2()
    noise_model = NoiseModel.from_backend(fake_backend)
    aer_simulator = AerSimulator(noise_model=noise_model)
    pass_manager = generate_preset_pass_manager(
        optimization_level=1, backend=aer_simulator
    )

    estimator = AerEstimatorV2(
        options={"backend_options": {"noise_model": noise_model}}
    )
    return estimator, pass_manager


def _expect_energy(estimator, circuit, observable, parameters):
    op = observable
    if getattr(circuit, "layout", None) is not None:
        op = observable.apply_layout(circuit.layout)
    evs = np.asarray(
        estimator.run([(circuit, op, parameters)]).result()[0].data.evs,
        dtype=float,
    ).reshape(-1)
    return float(np.real(evs[0]))


def _quantum_variance_std(estimator, circuit, observable, parameters, h_mean):
    # Quantum std of H in this state: sqrt(max(0, <H^2> - <H>^2)).
    h2 = _expect_energy(
        estimator, circuit, observable @ observable, parameters
    )
    return float(np.sqrt(max(0.0, h2 - h_mean**2)))


def _occupation_stds(estimator, circuit, parameters, n, next_pow2):
    # Occupation std per component: sqrt(p (1 - p)) for basis outcome probability p.
    stds = np.zeros(n, dtype=float)
    for j in range(n):
        proj = np.zeros((next_pow2, next_pow2), dtype=float)
        proj[j, j] = 1.0
        p_j = _expect_energy(
            estimator, circuit, SparsePauliOp.from_operator(proj), parameters
        )
        p_j = min(1.0, max(0.0, p_j))
        stds[j] = float(np.sqrt(max(0.0, p_j - p_j * p_j)))
    return stds


def run_vqd_eigensolver(flat_matrix, n, k=None, use_noise=False):
    """
    Takes a flat matrix of size n*n, and returns the lowest k eigenvalues
    and matching eigenvectors using VQD.

    With use_noise=True, only the energy estimator is noisy (FakeManila);
    VQD still recognizes distinct states perfectly.
    Expect worse eigenvalues, especially for k > 1.
    """
    if k is None:
        k = n
    if k < 1 or k > n:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")

    observable, next_pow2 = _matrix_to_observable(flat_matrix, n)
    num_qubits = int(np.log2(next_pow2))

    ansatz = EfficientSU2(num_qubits, reps=1, entanglement="linear")
    estimator, pass_manager = _make_estimator(use_noise)
    # Keep exact fidelity even under noise: overlaps stay exact; only energy
    # is FakeManila-noisy (same demo style as QAOA).
    fidelity = _ExactStatevectorFidelity()
    if use_noise:
        optimizer = COBYLA(maxiter=50)
    else:
        optimizer = SLSQP(maxiter=1000, ftol=1e-9)

    # Overlap weights on the scale of the original matrix, not the padding.
    beta = 10.0 * max(
        1.0, float(np.max(np.abs(np.array(flat_matrix, dtype=float))))
    )
    betas = np.full(k, beta)
    rng = np.random.default_rng(0)
    if k == 1:
        initial_points = np.zeros(ansatz.num_parameters)
    else:
        initial_points = [np.zeros(ansatz.num_parameters)]
        for _ in range(k - 1):
            initial_points.append(
                rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
            )

    if use_noise:
        vqd = VQD(
            estimator,
            fidelity,
            ansatz,
            optimizer,
            k=k,
            betas=betas,
            transpiler=pass_manager,
        )
    else:
        vqd = VQD(estimator, fidelity, ansatz, optimizer, k=k, betas=betas)
    vqd.initial_point = initial_points

    result = vqd.compute_eigenvalues(observable)

    eigenvalues = [float(np.real(e)) for e in result.eigenvalues]
    eigenvectors = np.zeros((n, n), dtype=float)
    for i in range(k):
        optimal_circuit = result.optimal_circuits[i]
        optimal_parameters = result.optimal_points[i]
        eigenvectors[:, i] = _statevector_to_real_eigenvector(
            optimal_circuit, optimal_parameters, n
        )

    uq_values = [0.0] * len(eigenvalues)
    uq_vectors = np.zeros((n, n), dtype=float)
    for i in range(k):
        optimal_circuit = result.optimal_circuits[i]
        optimal_parameters = result.optimal_points[i]
        uq_values[i] = _quantum_variance_std(
            estimator,
            optimal_circuit,
            observable,
            optimal_parameters,
            eigenvalues[i],
        )
        uq_vectors[:, i] = _occupation_stds(
            estimator,
            optimal_circuit,
            optimal_parameters,
            n,
            next_pow2,
        )

    return (
        eigenvalues,
        eigenvectors.ravel().tolist(),
        uq_values,
        uq_vectors.ravel().tolist(),
    )


def run_qaoa_eigensolver(flat_matrix, n, use_noise=False, reps=3):
    """
    Takes a flat matrix of size n*n, and returns the ground eigenvalue,
    and matching eigenvector using QAOA.
    """
    observable, next_pow2 = _matrix_to_observable(flat_matrix, n)
    ansatz = QAOAAnsatz(observable, reps=reps)
    estimator, pass_manager = _make_estimator(use_noise)
    if use_noise:
        optimizer = COBYLA(maxiter=50)
        vqe = VQE(estimator, ansatz, optimizer, transpiler=pass_manager)
    else:
        optimizer = SLSQP(maxiter=1000, ftol=1e-9)
        vqe = VQE(estimator, ansatz, optimizer)
    rng = np.random.default_rng(2)
    vqe.initial_point = rng.uniform(0, np.pi, ansatz.num_parameters)

    result = vqe.compute_minimum_eigenvalue(observable)

    optimal_circuit = result.optimal_circuit
    optimal_parameters = result.optimal_point

    eigenvalues = [float(np.real(result.eigenvalue))]
    eigenvectors = np.zeros((n, n), dtype=float)
    eigenvectors[:, 0] = _statevector_to_real_eigenvector(
        optimal_circuit, optimal_parameters, n
    )

    uq_values = [0.0] * len(eigenvalues)
    uq_vectors = np.zeros((n, n), dtype=float)
    uq_values[0] = _quantum_variance_std(
        estimator,
        optimal_circuit,
        observable,
        optimal_parameters,
        eigenvalues[0],
    )
    uq_vectors[:, 0] = _occupation_stds(
        estimator,
        optimal_circuit,
        optimal_parameters,
        n,
        next_pow2,
    )

    return (
        eigenvalues,
        eigenvectors.ravel().tolist(),
        uq_values,
        uq_vectors.ravel().tolist(),
    )


# Quick Test
if __name__ == "__main__":
    # The same 3x3 matrix from the C++ test file
    test_matrix = [3.0, 5.0, 2.0, 5.0, 1.0, 3.0, 2.0, 3.0, 2.0]

    print("=== VQD (noiseless) ===")
    values, vectors, vqd_uq_v, vqd_uq_vec = run_vqd_eigensolver(test_matrix, 3)
    print(f"Eigenvalues: {values}")
    print(f"Eigenvectors: {vectors}")
    print(f"uq_values: {vqd_uq_v}")
    print(f"uq_vectors: {vqd_uq_vec}")

    try:
        print("\n=== VQD (noise) ===")
        (
            vqd_noisy_values,
            vqd_noisy_vectors,
            vqd_noisy_uq_v,
            vqd_noisy_uq_vec,
        ) = run_vqd_eigensolver(test_matrix, 3, use_noise=True)
        print(f"Eigenvalues: {vqd_noisy_values}")
        print(f"Eigenvectors: {vqd_noisy_vectors}")
        print(f"uq_values: {vqd_noisy_uq_v}")
        print(f"uq_vectors: {vqd_noisy_uq_vec}")
    except ImportError as exc:
        print(f"Skipping noisy VQD demo: {exc}")

    print("\n=== QAOA (noiseless) ===")
    qaoa_values, qaoa_vectors, qaoa_uq_v, qaoa_uq_vec = run_qaoa_eigensolver(
        test_matrix, 3
    )
    print(f"Eigenvalue: {qaoa_values}")
    print(f"Eigenvector: {qaoa_vectors}")
    print(f"uq_values: {qaoa_uq_v}")
    print(f"uq_vectors: {qaoa_uq_vec}")

    try:
        print("\n=== QAOA (noise) ===")
        (
            qaoa_noisy_values,
            qaoa_noisy_vectors,
            noisy_uq_v,
            noisy_uq_vec,
        ) = run_qaoa_eigensolver(test_matrix, 3, use_noise=True)
        print(f"Eigenvalue: {qaoa_noisy_values}")
        print(f"Eigenvector: {qaoa_noisy_vectors}")
        print(f"uq_values: {noisy_uq_v}")
        print(f"uq_vectors: {noisy_uq_vec}")
    except ImportError as exc:
        print(f"Skipping noisy QAOA demo: {exc}")
