import numpy as np
import pytest
from DistributedPython.ProbabilityOfCollision.Utils.CheckAndResizePosVel import CheckAndResizePosVel

def test_single_vector_input():
    r = np.array([1, 2, 3])
    v = np.array([4, 5, 6])
    numR, v_out = CheckAndResizePosVel(r, v)
    assert numR == 1
    assert np.array_equal(v_out, np.array([[4, 5, 6]]))

def test_nx3_input_matching():
    r = np.array([[1, 2, 3], [7, 8, 9]])
    v = np.array([[4, 5, 6], [10, 11, 12]])
    numR, v_out = CheckAndResizePosVel(r, v)
    assert numR == 2
    assert np.array_equal(v_out, v)

def test_nx3_input_resize_v():
    r = np.array([[1, 2, 3], [7, 8, 9]])
    v = np.array([4, 5, 6])
    numR, v_out = CheckAndResizePosVel(r, v)
    assert numR == 2
    expected = np.array([[4, 5, 6], [4, 5, 6]])
    assert np.array_equal(v_out, expected)

    v = np.array([[4, 5, 6]])
    numR, v_out = CheckAndResizePosVel(r, v)
    assert numR == 2
    assert np.array_equal(v_out, expected)

def test_invalid_r_columns():
    r = np.array([[1, 2]])
    v = np.array([4, 5, 6])
    with pytest.raises(ValueError, match='r matrix must have 3 columns!'):
        CheckAndResizePosVel(r, v)

def test_invalid_v_columns():
    r = np.array([1, 2, 3])
    v = np.array([[4, 5]])
    with pytest.raises(ValueError, match='v matrix must have 3 columns!'):
        CheckAndResizePosVel(r, v)

def test_mismatch_rows():
    r = np.array([[1, 2, 3], [7, 8, 9]])
    v = np.array([[4, 5, 6], [10, 11, 12], [13, 14, 15]])
    with pytest.raises(ValueError, match='v matrix cannot be resized to match r matrix'):
        CheckAndResizePosVel(r, v)
