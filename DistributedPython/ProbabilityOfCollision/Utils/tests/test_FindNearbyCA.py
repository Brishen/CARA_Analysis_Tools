import numpy as np
import pytest
from DistributedPython.ProbabilityOfCollision.Utils.FindNearbyCA import FindNearbyCA

def test_linear_motion_zero_rel_vel():
    # Same velocity, different position
    r1 = np.array([0, 0, 0])
    v1 = np.array([1, 0, 0])
    r2 = np.array([10, 0, 0])
    v2 = np.array([1, 0, 0])
    X1 = np.concatenate((r1, v1))
    X2 = np.concatenate((r2, v2))

    dTCA, X1CA, X2CA = FindNearbyCA(X1, X2)
    assert np.isnan(dTCA)
    assert np.array_equal(X1CA[0:3], r1)
    assert np.array_equal(X2CA[0:3], r2)

def test_linear_motion_simple_case():
    # Object 1 at (0,0,0) moving (1,0,0)
    # Object 2 at (10,1,0) moving (-1,0,0)
    # Collision course in x, offset in y.
    # Relative velocity (-2, 0, 0).
    # Relative position (10, 1, 0).
    # TCA should be when x-distance is minimized (0).
    # x1(t) = t, x2(t) = 10 - t.
    # t = 10 - t => 2t = 10 => t = 5.

    r1 = np.array([0, 0, 0])
    v1 = np.array([1, 0, 0])
    r2 = np.array([10, 1, 0])
    v2 = np.array([-1, 0, 0])
    X1 = np.concatenate((r1, v1))
    X2 = np.concatenate((r2, v2))

    dTCA, X1CA, X2CA = FindNearbyCA(X1, X2)
    assert dTCA == 5.0

    # Check positions at TCA
    # X1: (5, 0, 0)
    # X2: (5, 1, 0)
    assert np.allclose(X1CA[0:3], np.array([5, 0, 0]))
    assert np.allclose(X2CA[0:3], np.array([5, 1, 0]))

def test_invalid_motion_mode():
    X1 = np.zeros(6)
    X2 = np.zeros(6)
    with pytest.raises(ValueError, match='Invalid motion mode'):
        FindNearbyCA(X1, X2, MotionMode='INVALID')

def test_twobody_not_implemented():
    X1 = np.zeros(6)
    X2 = np.zeros(6)
    with pytest.raises(NotImplementedError):
        FindNearbyCA(X1, X2, MotionMode='TWOBODY')

def test_invalid_input_size():
    X1 = np.zeros(5)
    X2 = np.zeros(6)
    with pytest.raises(ValueError, match='X1 and X2 must be 6-element vectors'):
        FindNearbyCA(X1, X2)
