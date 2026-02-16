import numpy as np
import pytest
from DistributedPython.ProbabilityOfCollision.Utils.CheckAndResizeCov import CheckAndResizeCov

def test_nx9_input_matching():
    numR = 2
    cov = np.zeros((2, 9))
    cov_out = CheckAndResizeCov(numR, cov)
    assert np.array_equal(cov_out, cov)

def test_nx9_input_mismatch():
    numR = 3
    cov = np.zeros((2, 9))
    with pytest.raises(ValueError, match='2D Covariance cannot be resized to match r matrix'):
        CheckAndResizeCov(numR, cov)

def test_3x3_input_resize():
    numR = 2
    cov = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # Flattened: 1, 2, 3, 4, 5, 6, 7, 8, 9
    expected = np.tile(np.arange(1, 10), (numR, 1))
    cov_out = CheckAndResizeCov(numR, cov)
    assert np.array_equal(cov_out, expected)

def test_6x6_input_resize():
    numR = 2
    cov = np.zeros((6, 6))
    cov[0:3, 0:3] = np.arange(1, 10).reshape(3, 3)
    expected = np.tile(np.arange(1, 10), (numR, 1))
    cov_out = CheckAndResizeCov(numR, cov)
    assert np.array_equal(cov_out, expected)

def test_3D_nx3x3_input():
    numR = 2
    cov = np.zeros((numR, 3, 3))
    # Fill with some data
    cov[0] = np.arange(1, 10).reshape(3, 3)
    cov[1] = np.arange(10, 19).reshape(3, 3)

    expected = np.vstack([np.arange(1, 10), np.arange(10, 19)])

    cov_out = CheckAndResizeCov(numR, cov)
    assert np.array_equal(cov_out, expected)

def test_3D_nx6x6_input():
    numR = 2
    cov = np.zeros((numR, 6, 6))
    cov[0, 0:3, 0:3] = np.arange(1, 10).reshape(3, 3)
    cov[1, 0:3, 0:3] = np.arange(10, 19).reshape(3, 3)

    expected = np.vstack([np.arange(1, 10), np.arange(10, 19)])

    cov_out = CheckAndResizeCov(numR, cov)
    assert np.array_equal(cov_out, expected)

def test_invalid_shape_2D():
    numR = 2
    cov = np.zeros((2, 8))
    with pytest.raises(ValueError, match='2D Covariance matrix must have 9 columns or be a 3x3 or 6x6 matrix!'):
        CheckAndResizeCov(numR, cov)

def test_invalid_shape_3D():
    numR = 2
    cov = np.zeros((numR, 4, 4))
    with pytest.raises(ValueError, match='3D covariance matrix must be of size numRx3x3 or numRx6x6'):
        CheckAndResizeCov(numR, cov)

def test_invalid_ndim():
    numR = 2
    cov = np.zeros((2, 2, 2, 2))
    with pytest.raises(ValueError, match='Improperly sized covariance was detected'):
        CheckAndResizeCov(numR, cov)
