import pytest
import numpy as np
import sys
import os

# Add the repo root to sys.path so we can import DistributedPython
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from DistributedPython.ProbabilityOfCollision.Utils.Product3x3 import Product3x3

def test_diagonal():
    # Tests 1000 random 3x3 matrices
    np.random.seed(42)
    A = 2000 * np.random.rand(1000, 9) - 1000
    B = 2000 * np.random.rand(1000, 9) - 1000

    P = Product3x3(A, B)

    for i in range(1000):
        # M = reshape(P(i, :), 3, 3)';
        # P(i, :) is [p1 p2 p3 p4 p5 p6 p7 p8 p9]
        # MATLAB reshape(..., 3, 3) fills column-wise:
        # [p1 p4 p7]
        # [p2 p5 p8]
        # [p3 p6 p9]
        # Transpose ' makes it:
        # [p1 p2 p3]
        # [p4 p5 p6]
        # [p7 p8 p9]
        # This is exactly what numpy reshape(..., (3, 3)) does (row-major filling).

        M = P[i, :].reshape(3, 3)

        # T = reshape(A(i, :), 3, 3)' * reshape(B(i, :), 3, 3)';
        # Similarly, A[i, :].reshape(3, 3) is matrix A_mat
        # B[i, :].reshape(3, 3) is matrix B_mat

        A_mat = A[i, :].reshape(3, 3)
        B_mat = B[i, :].reshape(3, 3)

        # T = A_mat * B_mat (matrix multiplication)
        T = np.matmul(A_mat, B_mat)

        # Verify equal
        np.testing.assert_allclose(M, T, rtol=1e-10, atol=1e-10)
