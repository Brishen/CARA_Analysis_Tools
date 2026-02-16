import unittest
import numpy as np
import sys
import os

# Add the repository root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from DistributedPython.ProbabilityOfCollision.Utils.CovRemEigValClip import CovRemEigValClip

class TestCovRemEigValClip(unittest.TestCase):

    def test_positive_definite_matrix(self):
        A = np.array([[2.0, 0.5], [0.5, 2.0]])
        # Eigenvalues are 1.5 and 2.5. Both > 0.

        res = CovRemEigValClip(A)

        self.assertEqual(res['PosDefStatus'], 1)
        self.assertFalse(res['ClipStatus'])
        # Sort eigenvalues for comparison as order might vary slightly but usually eigh sorts ascending
        self.assertTrue(np.allclose(np.sort(res['Lrem']), [1.5, 2.5]))
        self.assertAlmostEqual(res['Adet'], 3.75)
        self.assertTrue(np.allclose(res['Arem'], A))

        # Check Ainv
        Ainv_expected = np.linalg.inv(A)
        self.assertTrue(np.allclose(res['Ainv'], Ainv_expected))

    def test_non_positive_definite_matrix(self):
        # A matrix with eigenvalues 3 and -1
        # [1, 2; 2, 1] has eigenvalues 3 and -1
        A_npd = np.array([[1.0, 2.0], [2.0, 1.0]])

        res = CovRemEigValClip(A_npd, Lclip=0.0)

        self.assertEqual(res['PosDefStatus'], -1)
        self.assertTrue(res['ClipStatus'])

        # Lrem should have 0 instead of -1.
        expected_Lrem = [0.0, 3.0]
        self.assertTrue(np.allclose(np.sort(res['Lrem']), expected_Lrem))

        # Arem calculation
        # v1 (for -1) is [1/sqrt(2), -1/sqrt(2)]
        # v2 (for 3) is [1/sqrt(2), 1/sqrt(2)]
        # Arem = 0 * v1*v1.T + 3 * v2*v2.T
        # v2*v2.T = [[0.5, 0.5], [0.5, 0.5]]
        # Arem = [[1.5, 1.5], [1.5, 1.5]]

        expected_Arem = np.array([[1.5, 1.5], [1.5, 1.5]])
        self.assertTrue(np.allclose(res['Arem'], expected_Arem))

    def test_clipping_threshold(self):
        A = np.array([[1.0, 0.0], [0.0, 0.1]])
        Lclip = 0.2

        res = CovRemEigValClip(A, Lclip=Lclip)

        self.assertTrue(res['ClipStatus'])
        self.assertTrue(np.allclose(np.sort(res['Lrem']), [0.2, 1.0]))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            CovRemEigValClip(np.array([1, 2])) # Not 2D

        with self.assertRaises(ValueError):
            CovRemEigValClip(np.array([[1, 2], [3, 4]]), Lclip=-1) # Negative Lclip

        # Non-square
        with self.assertRaises(ValueError):
            CovRemEigValClip(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_column_lraw_input(self):
        # Verify that inputting Lraw as (N, 1) column vector works correctly
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        Lraw = np.array([[1.0], [2.0]])
        Vraw = np.array([[0.0, 1.0], [1.0, 0.0]])
        # A = 1 * [0,1]'[0,1] + 2 * [1,0]'[1,0] = [0 0; 0 1] + [2 0; 0 0] = [2 0; 0 1]

        res = CovRemEigValClip(A, Lraw=Lraw, Vraw=Vraw)

        # Check Arem reconstruction
        self.assertTrue(np.allclose(res['Arem'], A))

        # Check Ainv
        # Ainv = [0.5 0; 0 1]
        expected_Ainv = np.array([[0.5, 0.0], [0.0, 1.0]])
        self.assertTrue(np.allclose(res['Ainv'], expected_Ainv))

if __name__ == '__main__':
    unittest.main()
