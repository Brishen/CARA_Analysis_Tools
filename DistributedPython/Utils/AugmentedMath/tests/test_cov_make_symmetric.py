import numpy as np
import pytest
from DistributedPython.Utils.AugmentedMath.cov_make_symmetric import cov_make_symmetric

def test_cov_make_symmetric_valid():
    # Symmetric matrix, should remain same
    C = np.array([[2., 1.], [1., 2.]])
    Csym = cov_make_symmetric(C)
    np.testing.assert_array_equal(C, Csym)

    # Non-symmetric matrix
    C = np.array([[2., 1.1], [0.9, 2.]])
    # Expected: (C+C.T)/2 = [[2, 1], [1, 2]]
    # Then triu(avg,0) + triu(avg,1).T
    # avg = [[2, 1], [1, 2]]
    # triu(avg, 0) = [[2, 1], [0, 2]]
    # triu(avg, 1) = [[0, 1], [0, 0]]
    # triu(avg, 1).T = [[0, 0], [1, 0]]
    # result = [[2, 1], [1, 2]]
    Csym = cov_make_symmetric(C)
    expected = np.array([[2., 1.], [1., 2.]])
    np.testing.assert_array_almost_equal(Csym, expected)

def test_cov_make_symmetric_invalid():
    with pytest.raises(ValueError, match="Array needs to be a 2D matrix"):
        cov_make_symmetric(np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="Matrix needs to be square"):
        cov_make_symmetric(np.array([[1, 2, 3], [4, 5, 6]]))
