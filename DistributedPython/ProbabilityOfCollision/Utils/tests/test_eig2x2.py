import pytest
import numpy as np
import sys
import os

# Add the repo root to sys.path so we can import DistributedPython
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from DistributedPython.ProbabilityOfCollision.Utils.eig2x2 import eig2x2

def test_diagonal():
    # Tests 1000 random 2x2 real diagonal matrices
    np.random.seed(42)
    a = 2000 * np.random.rand(1000) - 1000
    b = np.zeros(1000)
    d = 2000 * np.random.rand(1000) - 1000

    Araw = np.column_stack((a, b, d))

    V1, V2, L1, L2 = eig2x2(Araw)

    for i in range(1000):
        # [V, D] = eig([a(i), b(i); b(i), d(i)], 'vector');
        matrix = np.array([[a[i], b[i]], [b[i], d[i]]])
        D, V = np.linalg.eigh(matrix)

        # [D, idx] = sort(D, 'descend');
        # eigh returns ascending.
        # So D[0] is smallest, D[1] is largest.
        # L1 is largest, L2 is smallest.

        expected_L1 = D[1]
        expected_L2 = D[0]

        # Verify eigenvalues
        assert np.isclose(L1[i], expected_L1, rtol=1e-12)
        assert np.isclose(L2[i], expected_L2, rtol=1e-12)

        # Verify eigenvectors
        # Eigenvectors can be +/-
        # V1 corresponds to L1 (largest) -> V[:, 1]
        # V2 corresponds to L2 (smallest) -> V[:, 0]

        # V1[i] vs V[:, 1]
        # Check if vectors are parallel (abs dot product close to 1) or abs diff close to 0

        # Since these are diagonal matrices, eigenvectors are [1, 0] and [0, 1]
        # But order might swap if eigenvalues swap.

        dot1 = np.dot(V1[i], V[:, 1])
        assert np.isclose(abs(dot1), 1.0, atol=1e-10)

        dot2 = np.dot(V2[i], V[:, 0])
        assert np.isclose(abs(dot2), 1.0, atol=1e-10)

def test_zero_matrix():
    # Tests 2x2 zero matrix
    Araw = np.array([[0, 0, 0]])
    V1, V2, L1, L2 = eig2x2(Araw)

    assert np.allclose(L1, 0)
    assert np.allclose(L2, 0)
    # V1 = [1, 0], V2 = [0, 1] (default for d <= a)
    # a=0, d=0. d <= a is true.
    assert np.allclose(V1, [1, 0])
    assert np.allclose(V2, [0, 1])

def test_random_matrices():
    # Tests 1000 random 2x2 real symmetric matrices
    np.random.seed(43)
    a = 2000 * np.random.rand(1000) - 1000
    b = 2000 * np.random.rand(1000) - 1000
    d = 2000 * np.random.rand(1000) - 1000

    Araw = np.column_stack((a, b, d))

    V1, V2, L1, L2 = eig2x2(Araw)

    for i in range(1000):
        matrix = np.array([[a[i], b[i]], [b[i], d[i]]])
        D, V = np.linalg.eigh(matrix)

        expected_L1 = D[1]
        expected_L2 = D[0]

        assert np.isclose(L1[i], expected_L1, rtol=1e-5, atol=1e-5)
        assert np.isclose(L2[i], expected_L2, rtol=1e-5, atol=1e-5)

        # Check eigenvectors direction (dot product should be +/- 1)
        # V1 vs V[:, 1]
        dot1 = np.dot(V1[i], V[:, 1])
        assert np.isclose(abs(dot1), 1.0, atol=1e-5)

        # V2 vs V[:, 0]
        dot2 = np.dot(V2[i], V[:, 0])
        assert np.isclose(abs(dot2), 1.0, atol=1e-5)

def test_small_off_diagonal_matrices():
    # Tests 1000 random 2x2 real symmetric matrices with small off-diagonal values
    np.random.seed(44)
    a = 2000 * np.random.rand(1000) - 1000
    b = 2e-2 * np.random.rand(1000) - 1e-2
    d = 2000 * np.random.rand(1000) - 1000

    Araw = np.column_stack((a, b, d))

    V1, V2, L1, L2 = eig2x2(Araw)

    for i in range(1000):
        matrix = np.array([[a[i], b[i]], [b[i], d[i]]])
        D, V = np.linalg.eigh(matrix)

        expected_L1 = D[1]
        expected_L2 = D[0]

        assert np.isclose(L1[i], expected_L1, rtol=1e-5, atol=1e-5)
        assert np.isclose(L2[i], expected_L2, rtol=1e-5, atol=1e-5)

        dot1 = np.dot(V1[i], V[:, 1])
        assert np.isclose(abs(dot1), 1.0, atol=1e-5)

        dot2 = np.dot(V2[i], V[:, 0])
        assert np.isclose(abs(dot2), 1.0, atol=1e-5)

def test_default_covariance():
    a = 4.06806226869326435234562435e+15
    b = 251651.25
    d = 4.06806226869326435234562435e+15

    Araw = np.array([[a, b, d]])

    V1, V2, L1, L2 = eig2x2(Araw)

    matrix = np.array([[a, b], [b, d]])
    D, V = np.linalg.eigh(matrix)

    expected_L1 = D[1]
    expected_L2 = D[0]

    assert np.isclose(L1[0], expected_L1)
    assert np.isclose(L2[0], expected_L2)

    dot1 = np.dot(V1[0], V[:, 1])
    assert np.isclose(abs(dot1), 1.0)

    dot2 = np.dot(V2[0], V[:, 0])
    assert np.isclose(abs(dot2), 1.0)
