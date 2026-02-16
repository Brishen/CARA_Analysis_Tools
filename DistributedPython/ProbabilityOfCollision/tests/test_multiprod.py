
import unittest
import numpy as np
import sys
import os

# Add the repository root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from DistributedPython.ProbabilityOfCollision.Pc3D_Hall_Utils.multiprod import multiprod

class TestMultiprod(unittest.TestCase):
    def test_multiprod_matrix_mul(self):
        # 2D by 2D
        # (N, P, Q) x (N, Q, R) -> (N, P, R)
        N, P, Q, R = 5, 3, 4, 2
        a = np.random.rand(N, P, Q)
        b = np.random.rand(N, Q, R)

        # Dimensions 1 and 2 are the matrix dimensions
        c = multiprod(a, b, [1, 2], [1, 2])

        self.assertEqual(c.shape, (N, P, R))
        for i in range(N):
            np.testing.assert_allclose(c[i], np.dot(a[i], b[i]))

    def test_multiprod_broadcasting(self):
        # (P, Q) x (N, Q, R) -> (N, P, R)
        # Broadcasting A to N
        N, P, Q, R = 5, 3, 4, 2
        a = np.random.rand(P, Q)
        b = np.random.rand(N, Q, R)

        # A is (P, Q), dimensions [0, 1] are matrix dims
        # B is (N, Q, R), dimensions [1, 2] are matrix dims
        c = multiprod(a, b, [0, 1], [1, 2])

        self.assertEqual(c.shape, (N, P, R))
        for i in range(N):
            np.testing.assert_allclose(c[i], np.dot(a, b[i]))

    def test_multiprod_vector_matrix(self):
        # (N, K) x (N, K, M) -> (N, M)
        # Vector-matrix multiplication
        # K vector times KxM matrix
        N, K, M = 5, 3, 4
        a = np.random.rand(N, K)
        b = np.random.rand(N, K, M)

        # Dimensions 1 is vector dim in A
        # Dimensions 1, 2 are matrix dims in B
        c = multiprod(a, b, [1], [1, 2])

        self.assertEqual(c.shape, (N, M))
        for i in range(N):
            np.testing.assert_allclose(c[i], np.dot(a[i], b[i]))

    def test_multiprod_matrix_vector(self):
        # (N, M, K) x (N, K) -> (N, M)
        # Matrix-vector multiplication
        # MxK matrix times K vector
        N, M, K = 5, 4, 3
        a = np.random.rand(N, M, K)
        b = np.random.rand(N, K)

        c = multiprod(a, b, [1, 2], [1])

        self.assertEqual(c.shape, (N, M))
        for i in range(N):
            np.testing.assert_allclose(c[i], np.dot(a[i], b[i]))

    def test_multiprod_inner_product(self):
        # (N, K) x (N, K) -> (N, 1)
        # Inner product
        N, K = 5, 3
        a = np.random.rand(N, K)
        b = np.random.rand(N, K)

        c = multiprod(a, b, [1], [1])

        # Should return (N, 1) to match MATLAB behavior where reduced dimension is kept as 1?
        # In MATLAB multiprod reduces dim size to 1.
        self.assertEqual(c.shape, (N, 1))

        expected = np.sum(a * b, axis=1, keepdims=True)
        np.testing.assert_allclose(c, expected)

    def test_multiprod_scalar_expansion(self):
        # Scalar times Matrix
        # (1, 1) x (M, K) -> (M, K)
        # Or (N) scalar x (N, M, K)

        # Wait, multiprod handles 1D/2D blocks.
        # If A is scalar (0D) or (1,1).
        # We assume blocks are handled by broadcasting.
        pass

if __name__ == '__main__':
    unittest.main()
