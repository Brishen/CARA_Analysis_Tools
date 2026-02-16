import numpy as np
from scipy.special import erf, erfc

def erf_vec_dif(a, b):
    """
    Calculate the difference d = erf(a) - erf(b), but use erfc for cases of
    large positive or negative values of a and b, which provides improved
    accuracy.

    Args:
        a (np.ndarray): Input array a
        b (np.ndarray): Input array b

    Returns:
        np.ndarray: Difference erf(a) - erf(b)
    """
    a = np.asarray(a)
    b = np.asarray(b)

    # Broadcast inputs to common shape
    try:
        a_bc, b_bc = np.broadcast_arrays(a, b)
    except ValueError:
        raise ValueError("Inputs 'a' and 'b' must be broadcastable to the same shape.")

    # Large measure for argument of erf
    large = 3.0

    # Initialize result array
    d = np.full_like(a_bc, np.nan, dtype=np.float64)

    # Determine masks for different cases
    minab = np.minimum(a_bc, b_bc)
    maxab = np.maximum(a_bc, b_bc)

    set1 = minab > large
    set2 = maxab < -large
    set3 = ~(set1 | set2)

    # Case 1: Large positive a & b values, use erfc for better accuracy
    # d = erf(a) - erf(b) = (1 - erfc(a)) - (1 - erfc(b)) = erfc(b) - erfc(a)
    if np.any(set1):
        d[set1] = erfc(b_bc[set1]) - erfc(a_bc[set1])

    # Case 2: Large negative a & b values, use erfc with negated arguments
    # d = erf(a) - erf(b) = (erfc(-a) - 1) - (erfc(-b) - 1) = erfc(-a) - erfc(-b)
    if np.any(set2):
        d[set2] = erfc(-a_bc[set2]) - erfc(-b_bc[set2])

    # Case 3: Use erf for all other cases
    if np.any(set3):
        d[set3] = erf(a_bc[set3]) - erf(b_bc[set3])

    return d
