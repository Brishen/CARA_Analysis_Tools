import numpy as np

def cov_make_symmetric(C):
    """
    Make a covariance matrix diagonally symmetric if required.

    Parameters
    ----------
    C : ndarray
        Input covariance matrix, must be [NxN]

    Returns
    -------
    Csym : ndarray
        Symmetrized version of covariance matrix [NxN]
    """
    # Check for bad covariance matrix input
    if C.ndim != 2:
        raise ValueError("Array needs to be a 2D matrix to make symmetric.")

    if C.shape[0] != C.shape[1]:
        raise ValueError("Matrix needs to be square to make symmetric.")

    # Calculate transpose
    Ct = C.T

    # Check existing status of diagonal symmetry
    if np.array_equal(C, Ct):
        # Original matrix is already diagonally symmetric
        Csym = C.copy()
    else:
        # Average out any off-diagonal asymmetries
        Csym = (C + Ct) / 2.0

        # Reflect about diagonal to ensure diagonal symmetry absolutely
        # Csym = triu(Csym,0)+triu(Csym,1)'; in Matlab
        Csym = np.triu(Csym, 0) + np.triu(Csym, 1).T

    return Csym
