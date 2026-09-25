<!--
  ~ Copyright 2026 QHARM
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~ http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# EigenBridge

There are two quantum solvers. VQD can return several eigenvalues; QAOA returns only the ground state.

VQD -> `run_vqd_eigensolver(matrix, k, use_noise=false)` -> lowest `k` eigenvalues (`k` defaults to all)
QAOA -> `run_qaoa_eigensolver(matrix, use_noise=false)` -> ground state only

**Uncertainties (`uq_values` / `uq_vectors`)**
- For every VQD/QAOA path (noiseless and noisy), at each final optimized circuit/state,
  `uq_values[i] = sqrt(max(0, ⟨H²⟩ − ⟨H⟩²))`, where ⟨H⟩ is the reported eigenvalue and ⟨H²⟩
  comes from the same estimator on `H @ H` at the same parameters.
  This is the standard deviation of a single measurement of H in the prepared state
  (intrinsic quantum / variational residual variance).
  Exact eigenstates give σ ≈ 0; approximate ansatz states give nonzero σ even without noise.
- `uq_vectors` uses the same layout as the returned eigenvectors (column `i` = mode `i`).
  For each physical basis index `j`,
  `uq_vectors[j, i] = sqrt(p_j (1 − p_j))` where `p_j = ⟨|j⟩⟨j|⟩` is evaluated with the
  **same estimator / circuit / parameters** as the energy (FakeManila when `use_noise=true`).
  This is the std of a single-shot computational-basis occupation measurement for component `j`.
  Caveat: it can be > 0 even for an exact delocalized eigenstate (unlike σ_H, which is 0 for
  exact energy eigenstates). It is **not** “distance to the classical eigenvector.”
  Unused eigenvector columns stay 0. Reported eigenvectors still come from the ideal
  statevector of the optimized circuit; only the UQ uses the estimator/noise path.
- Non–power-of-2 dimensions pad to the next power of two and place
  `penalty = max(1.0, Gershgorin row-sum bound)` on unused diagonal entries so pad
  eigenvalues stay outside the original spectrum while avoiding the old 10× entry-scale
  inflation of ⟨H²⟩ on small problems.
- Noisy VQD keeps exact statevector overlaps; only the energy / projector estimator uses
  FakeManila noise (demo). Expect worse eigenvalues, especially for higher states.

Unit tests check that reported `uq_values` / filled `uq_vectors` entries are finite and
nonnegative (filled columns not identically zero on the demo matrix), and that on the
demo 3×3 matrix the ground-state interval `[⟨H⟩−σ, ⟨H⟩+σ]` contains the classical ground
eigenvalue for noisy paths. They do not require noisy eigenvalues to match the classical answer.
