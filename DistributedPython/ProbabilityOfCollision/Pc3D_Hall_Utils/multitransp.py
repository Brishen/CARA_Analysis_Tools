
import numpy as np

def multitransp(a, dim=0):
    """
    Transposing arrays of matrices.
    Equivalent to MATLAB's multitransp(a, dim+1) if dim is 0-based.

    Swaps dimensions dim and dim+1.

    Parameters
    ----------
    a : array_like
        Input array.
    dim : int, optional
        The first dimension to swap. The second dimension is dim+1.
        Default is 0.

    Returns
    -------
    b : ndarray
        Transposed array.
    """
    return np.swapaxes(a, dim, dim+1)
