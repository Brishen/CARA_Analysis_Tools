import pytest
import numpy as np
import sys
import os

# Add the repo root to sys.path so we can import DistributedPython
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from DistributedPython.ProbabilityOfCollision.Utils.CovRemEigValClip2x2 import CovRemEigValClip2x2

def test_no_clipping_needed():
    # Case where eigenvalues are already positive
    # Matrix [2 0; 0 3] -> eigenvalues 2, 3
    Araw = np.array([[2, 0, 3]])
    ClipStatus, Arem = CovRemEigValClip2x2(Araw, Lclip=0)

    assert ClipStatus[0] == False
    assert np.allclose(Arem, Araw)

def test_clipping_needed():
    # Case where one eigenvalue is negative
    # Matrix [-1 0; 0 2] -> eigenvalues -1, 2
    # Lclip = 0
    Araw = np.array([[-1, 0, 2]])
    ClipStatus, Arem = CovRemEigValClip2x2(Araw, Lclip=0)

    assert ClipStatus[0] == True

    # Expected remediated matrix:
    # Eigenvalues should be clipped to 0. So 0 and 2.
    # Matrix should be reconstructed with eigenvalues 0 and 2.
    # Original eigenvectors: [1, 0] (for -1) and [0, 1] (for 2)
    # Remediated: [1, 0] has eigval 0. [0, 1] has eigval 2.
    # Matrix = 0*[1 0]'*[1 0] + 2*[0 1]'*[0 1] = [0 0; 0 2]

    expected_Arem = np.array([[0, 0, 2]])
    assert np.allclose(Arem, expected_Arem)

def test_clipping_both_needed():
    # Matrix [-2 0; 0 -3] -> eigenvalues -2, -3
    # Lclip = 0
    Araw = np.array([[-2, 0, -3]])
    ClipStatus, Arem = CovRemEigValClip2x2(Araw, Lclip=0)

    assert ClipStatus[0] == True

    # Expected: eigenvalues 0, 0. Matrix zero.
    expected_Arem = np.array([[0, 0, 0]])
    assert np.allclose(Arem, expected_Arem)

def test_clipping_with_rotation():
    # Matrix with rotation.
    # [1 2; 2 1]. Eigenvalues:
    # T = 2, D = 1-4 = -3.
    # L = (2 +/- sqrt(4 - 4(-3)))/2 = (2 +/- 4)/2 => 3, -1.
    # Eigenvectors:
    # for 3: [1 1]/sqrt(2)
    # for -1: [1 -1]/sqrt(2)

    # Clip -1 to 0. Lclip = 0.
    # New eigenvalues: 3, 0.
    # Reconstruct:
    # 3 * [1 1]'[1 1]/2 + 0 * ...
    # = 1.5 * [1 1; 1 1]
    # = [1.5 1.5; 1.5 1.5]

    Araw = np.array([[1, 2, 1]])
    ClipStatus, Arem = CovRemEigValClip2x2(Araw, Lclip=0)

    assert ClipStatus[0] == True

    expected_Arem = np.array([[1.5, 1.5, 1.5]])
    assert np.allclose(Arem, expected_Arem)

def test_positive_lclip():
    # Lclip = 0.5
    # Matrix [-1 0; 0 2] -> eigenvalues -1, 2.
    # Clipped to 0.5, 2.
    # Matrix = 0.5 * [1 0]'[1 0] + 2 * [0 1]'[0 1] = [0.5 0; 0 2]

    Araw = np.array([[-1, 0, 2]])
    ClipStatus, Arem = CovRemEigValClip2x2(Araw, Lclip=0.5)

    assert ClipStatus[0] == True

    expected_Arem = np.array([[0.5, 0, 2]])
    assert np.allclose(Arem, expected_Arem)

def test_multiple_matrices():
    # Test vectorized behavior
    Araw = np.array([
        [2, 0, 3],   # OK
        [-1, 0, 2],  # Clip -> [0 0 2]
        [1, 2, 1]    # Clip -> [1.5 1.5 1.5]
    ])

    ClipStatus, Arem = CovRemEigValClip2x2(Araw, Lclip=0)

    assert np.array_equal(ClipStatus, [False, True, True])

    expected_Arem = np.array([
        [2, 0, 3],
        [0, 0, 2],
        [1.5, 1.5, 1.5]
    ])

    assert np.allclose(Arem, expected_Arem)

def test_array_lclip():
    # Test vectorized behavior with array Lclip
    # Mat 1: [2 0 3] -> No clip. Lclip=0
    # Mat 2: [-1 0 2] -> Clip to 0.5. Lclip=0.5. Result: [0.5 0 2]
    # Mat 3: [1 2 1] -> Clip to 1.0. Lclip=1.0. (Eigs: 3, -1 -> 3, 1)
    # Reconstruct Mat 3: 3*v1v1' + 1*v2v2'. v1=[1 1]/s2, v2=[1 -1]/s2
    # = 3/2 [1 1; 1 1] + 1/2 [1 -1; -1 1]
    # = [1.5 1.5; 1.5 1.5] + [0.5 -0.5; -0.5 0.5]
    # = [2 1; 1 2] -> [2 1 2]

    Araw = np.array([
        [2, 0, 3],
        [-1, 0, 2],
        [1, 2, 1]
    ])
    Lclip = np.array([0, 0.5, 1.0])

    ClipStatus, Arem = CovRemEigValClip2x2(Araw, Lclip=Lclip)

    assert np.array_equal(ClipStatus, [False, True, True])

    expected_Arem = np.array([
        [2, 0, 3],
        [0.5, 0, 2],
        [2, 1, 2]
    ])

    assert np.allclose(Arem, expected_Arem)
