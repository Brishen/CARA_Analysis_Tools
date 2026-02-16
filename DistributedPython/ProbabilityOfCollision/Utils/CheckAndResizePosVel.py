import numpy as np

def CheckAndResizePosVel(r, v):
    """
    CheckAndResizePosVel - Checks the sizes of the matrices and resizes v if needed.

    Args:
        r (np.ndarray): Position matrix [nx3]
        v (np.ndarray): Velocity matrix [mx3]

    Returns:
        tuple:
            numR (int): Number of rows in r
            v (np.ndarray): Resized velocity matrix [nx3]
    """
    r = np.asarray(r)
    v = np.asarray(v)

    # Handle 1D input (single vector)
    if r.ndim == 1:
        if r.size == 3:
            r = r.reshape(1, 3)
        else:
            raise ValueError('r matrix must have 3 columns!')

    if v.ndim == 1:
        if v.size == 3:
            v = v.reshape(1, 3)
        else:
             raise ValueError('v matrix must have 3 columns!')

    numR, rColumns = r.shape
    if rColumns != 3:
        raise ValueError('r matrix must have 3 columns!')

    numV, vColumns = v.shape
    if vColumns != 3:
        raise ValueError('v matrix must have 3 columns!')

    if numV != numR:
        if numV == 1:
            v = np.tile(v, (numR, 1))
        else:
            raise ValueError('v matrix cannot be resized to match r matrix')

    return numR, v
