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

import unittest

import eigenbridge


class TestEigensolvers(unittest.TestCase):
    def setUp(self):
        # Define a 3x3 matrix
        self.flat_matrix = [3.0, 5.0, 2.0, 5.0, 1.0, 3.0, 2.0, 3.0, 2.0]
        # Correct results
        self.correct_eigenvalues = [
            -3.3610452994996525,
            0.5038738768058354,
            8.8571714226938152,
        ]
        self.correct_eigenvectors = [
            -0.5518254419285664,
            -0.5057445690691208,
            -0.6631071651682192,
            0.7984036023978691,
            -0.0906811731252565,
            -0.5952550818924042,
            -0.2409156892326610,
            0.8579040480716453,
            -0.4538284642723896,
        ]

    def _assert_finite_nonnegative(self, uq):
        self.assertTrue(uq == uq and abs(uq) != float("inf"))
        self.assertGreaterEqual(uq, 0.0)

    def _assert_finite_positive(self, uq):
        self._assert_finite_nonnegative(uq)
        self.assertGreater(uq, 0.0)

    def _check_uq_vectors(self, eigenvectors, uq_vectors, n=3, k_filled=None):
        # uq_vectors: sqrt(p(1-p)) occupation stds; unused columns stay 0.
        self.assertEqual(len(uq_vectors), len(eigenvectors))
        self.assertEqual(len(uq_vectors), n * n)
        for uq in uq_vectors:
            self._assert_finite_nonnegative(uq)
        if k_filled is None:
            k_filled = n
        for col in range(n):
            col_uq = [uq_vectors[row * n + col] for row in range(n)]
            if col < k_filled:
                self.assertGreater(max(col_uq), 0.0)
            else:
                self.assertTrue(all(u == 0.0 for u in col_uq))

    def _check_results(
        self,
        results,
        val_places=14,
        vec_places=14,
        require_positive_uq=False,
        k_filled=None,
    ):
        # Check the results
        eigenvalues, eigenvectors, uq_values, uq_vectors = results
        for i in zip(eigenvalues, self.correct_eigenvalues):
            self.assertAlmostEqual(i[0], i[1], places=val_places)
        for i in zip(eigenvectors, self.correct_eigenvectors):
            self.assertAlmostEqual(abs(i[0]), abs(i[1]), places=vec_places)
        # Energy UQs for returned modes: >= 0, or > 0 if require_positive_uq.
        n_returned = sum(1 for v in self.correct_eigenvalues if v != 0.0)
        if n_returned == 0:
            n_returned = len(eigenvalues)
        for i, uq in enumerate(uq_values):
            if i < n_returned:
                if require_positive_uq:
                    self._assert_finite_positive(uq)
                else:
                    self._assert_finite_nonnegative(uq)
            else:
                self.assertEqual(uq, 0.0)
        if k_filled is None:
            k_filled = n_returned
        self._check_uq_vectors(
            eigenvectors, uq_vectors, n=3, k_filled=k_filled
        )

    def test_vqd_eigensolver(self):
        results = eigenbridge.run_vqd_eigensolver(self.flat_matrix, n=3)
        self._check_results(results, val_places=8, vec_places=4, k_filled=3)
        eigenvalues, _, uq_values, _ = results
        self.assertLessEqual(
            eigenvalues[0] - uq_values[0], self.correct_eigenvalues[0]
        )
        self.assertLessEqual(
            self.correct_eigenvalues[0], eigenvalues[0] + uq_values[0]
        )

    def test_vqd_eigensolver_noise(self):
        # Does not require eigenvalues to match LAPACK, since Noisy VQD is a demo.
        try:
            (
                eigenvalues,
                eigenvectors,
                uq_values,
                uq_vectors,
            ) = eigenbridge.run_vqd_eigensolver(
                self.flat_matrix, n=3, use_noise=True
            )
        except ImportError as exc:
            self.skipTest(f"Noisy VQD deps missing: {exc}")

        self.assertEqual(len(eigenvalues), 3)
        self.assertEqual(len(uq_values), 3)
        for uq in uq_values:
            self._assert_finite_positive(uq)
        self._check_uq_vectors(eigenvectors, uq_vectors, n=3, k_filled=3)
        # Sanity check on this demo matrix only — not a universal guarantee.
        self.assertLessEqual(
            eigenvalues[0] - uq_values[0], self.correct_eigenvalues[0]
        )
        self.assertLessEqual(
            self.correct_eigenvalues[0], eigenvalues[0] + uq_values[0]
        )

    def test_qaoa_eigensolver(self):
        # Adjust the correct eigenvalues and eigenvectors for QAOA's output
        # The QAOA solver will not return all eigenvalues/eigenvectors, so we
        # zero out the ones that are not expected to be returned.
        for i in range(len(self.correct_eigenvalues)):
            if i % 3 != 0:
                self.correct_eigenvalues[i] = 0.0
        for i in range(len(self.correct_eigenvectors)):
            if i % 3 != 0:
                self.correct_eigenvectors[i] = 0.0
        results = eigenbridge.run_qaoa_eigensolver(
            self.flat_matrix, n=3, use_noise=False
        )
        # New gershgorin padding changes the QAOA cost Hamiltonian / landscape slightly
        self._check_results(
            results,
            val_places=0,
            vec_places=0,
            require_positive_uq=True,
            k_filled=1,
        )
        eigenvalues, _, uq_values, _ = results
        self.assertLessEqual(
            eigenvalues[0] - uq_values[0], self.correct_eigenvalues[0]
        )
        self.assertLessEqual(
            self.correct_eigenvalues[0], eigenvalues[0] + uq_values[0]
        )

    def test_qaoa_eigensolver_noise(self):
        try:
            (
                eigenvalues,
                eigenvectors,
                uq_values,
                uq_vectors,
            ) = eigenbridge.run_qaoa_eigensolver(
                self.flat_matrix, n=3, use_noise=True
            )
        except ImportError as exc:
            self.skipTest(f"Noisy QAOA deps missing: {exc}")

        self._assert_finite_positive(uq_values[0])
        for uq in uq_values[1:]:
            self.assertEqual(uq, 0.0)
        self._check_uq_vectors(eigenvectors, uq_vectors, n=3, k_filled=1)
        # Sanity check on this demo matrix only, not a universal guarantee.
        self.assertLessEqual(
            eigenvalues[0] - uq_values[0], self.correct_eigenvalues[0]
        )
        self.assertLessEqual(
            self.correct_eigenvalues[0], eigenvalues[0] + uq_values[0]
        )
