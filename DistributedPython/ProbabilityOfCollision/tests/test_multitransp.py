
import unittest
import numpy as np
import sys
import os

# Add the repository root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from DistributedPython.ProbabilityOfCollision.Pc3D_Hall_Utils.multitransp import multitransp

class TestMultitransp(unittest.TestCase):
    def test_multitransp_default(self):
        # 3D array (2 matrices of 3x2)
        # Shape (2, 3, 2)
        # Matrices are along axes 1 and 2 (3x2)
        # multitransp(a, dim=1) swaps axes 1 and 2
        # Default dim=0 swaps axes 0 and 1

        # Let's match MATLAB default dim=1 (which corresponds to python dim=0 for swapaxes(0, 1)?)
        # MATLAB multitransp(A) -> multitransp(A, 1) -> swaps dim 1 and 2.
        # Python multitransp(A) -> multitransp(A, 0) -> swaps dim 0 and 1.

        # Test case: 2 matrices of 3x4.
        # A has shape (2, 3, 4).
        # We want to transpose the matrices.
        # But wait, usually in Python we have (N, P, Q).
        # If we want to transpose the matrices, we swap axis 1 and 2.
        # So multitransp(A, 1).

        a = np.random.rand(2, 3, 4)
        b = multitransp(a, 1)

        self.assertEqual(b.shape, (2, 4, 3))
        for i in range(2):
            np.testing.assert_array_equal(b[i], a[i].T)

    def test_multitransp_dim0(self):
        # Test swapping first two dimensions
        a = np.random.rand(3, 4, 5)
        b = multitransp(a, 0)

        self.assertEqual(b.shape, (4, 3, 5))
        np.testing.assert_array_equal(b, np.swapaxes(a, 0, 1))

    def test_multitransp_higher_dim(self):
        # 4D array
        a = np.random.rand(2, 3, 4, 5)
        # Swap axes 2 and 3
        b = multitransp(a, 2)

        self.assertEqual(b.shape, (2, 3, 5, 4))
        np.testing.assert_array_equal(b, np.swapaxes(a, 2, 3))

if __name__ == '__main__':
    unittest.main()
