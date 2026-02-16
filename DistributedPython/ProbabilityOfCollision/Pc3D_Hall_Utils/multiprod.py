
import numpy as np

def multiprod(a, b, axA=None, axB=None):
    """
    Multiplying 1-D or 2-D subarrays contained in two N-D arrays.

    Mimics MATLAB's multiprod behavior.

    Parameters
    ----------
    a : array_like
        Input array A.
    b : array_like
        Input array B.
    axA : list of int, optional
        The dimensions in A forming the matrix or vector.
        Default is [0, 1].
    axB : list of int, optional
        The dimensions in B forming the matrix or vector.
        Default is [0, 1].

    Returns
    -------
    c : ndarray
        The product array.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if axA is None:
        axA = [0, 1]
    if axB is None:
        axB = [0, 1]

    if np.isscalar(axA):
        axA = [axA]
    if np.isscalar(axB):
        axB = [axB]

    # Ensure lists
    axA = list(axA)
    axB = list(axB)

    # Handle negative indices
    ndimA = a.ndim
    ndimB = b.ndim

    axA = [d if d >= 0 else ndimA + d for d in axA]
    axB = [d if d >= 0 else ndimB + d for d in axB]

    # Move specified axes to the end
    a_moved = np.moveaxis(a, axA, [-len(axA) + i for i in range(len(axA))])
    b_moved = np.moveaxis(b, axB, [-len(axB) + i for i in range(len(axB))])

    # Handle vectors by expanding dims
    if len(axA) == 1:
        a_moved = np.expand_dims(a_moved, -2) # (..., 1, K)

    if len(axB) == 1:
        b_moved = np.expand_dims(b_moved, -1) # (..., K, 1)

    # Perform multiplication
    try:
        c_moved = np.matmul(a_moved, b_moved) # (..., rows, cols)
    except ValueError as e:
        # Fallback or better error message?
        # Typically broadcasting error or dimension mismatch
        raise ValueError(f"multiprod failed: {e}")

    # Determine result block axes handling
    if len(axA) == 2 and len(axB) == 2:
        # (P, Q) x (Q, R) -> (P, R). Keep both.
        result_ndim = 2
        source_axes = [-2, -1]
    elif len(axA) == 2 and len(axB) == 1:
        # (P, Q) x (Q, 1) -> (P, 1). Squeeze -1.
        c_moved = np.squeeze(c_moved, -1)
        result_ndim = 1
        source_axes = [-1]
    elif len(axA) == 1 and len(axB) == 2:
        # (1, Q) x (Q, R) -> (1, R). Squeeze -2.
        c_moved = np.squeeze(c_moved, -2)
        result_ndim = 1
        source_axes = [-1]
    elif len(axA) == 1 and len(axB) == 1:
        # (1, Q) x (Q, 1) -> (1, 1).
        # MATLAB returns size 1 dimension.
        c_moved = np.squeeze(c_moved, -1) # (..., 1)
        result_ndim = 1
        source_axes = [-1]
    else:
        raise ValueError("Only 1-D and 2-D blocks are supported.")

    # Move result axes to proper location
    # Heuristic: max(axA[0], axB[0])
    # Note: if axA or axB are not sorted, using [0] might be misleading,
    # but MATLAB multiprod usually expects contiguous ascending axes.
    dest_start = max(axA[0], axB[0])

    # Handle case where dest_start is out of bounds for the current shape (unlikely if broadcasting worked)
    # But np.moveaxis handles negative indices if needed.
    # We construct dest_axes based on result_ndim
    dest_axes = [dest_start + i for i in range(result_ndim)]

    c = np.moveaxis(c_moved, source_axes, dest_axes)
    return c
