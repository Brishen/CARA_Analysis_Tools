import numpy as np

def Product3x3(a, b):
    """
    Product3x3 - vectorized 3x3 matrix multiplication routine

    Args:
        a (np.ndarray): Left matrix [nx9]
        b (np.ndarray): Right matrix [nx9]

    Returns:
        np.ndarray: Matrix product [nx9]

    Notes:
        Inputs and outputs are formatted as row-major [row1, row2, row3]:
        [(1,1) (1,2) (1,3) (2,1) (2,2) (2,3) (3,1) (3,2) (3,3)]
    """
    # Ensure inputs are numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)

    # Check if inputs are 1D arrays (single matrix case) and reshape/expand if necessary
    # The original MATLAB code expects [nx9]. If n=1, it might be [1x9] or [9x1] in MATLAB.
    # Here we assume [nx9].

    # The MATLAB implementation uses indices corresponding to a row-major layout
    # (flattening row by row), which matches standard C/Python layout.

    # We will use column stacking to create the output array

    c1 = a[:,0]*b[:,0] + a[:,1]*b[:,3] + a[:,2]*b[:,6]
    c2 = a[:,0]*b[:,1] + a[:,1]*b[:,4] + a[:,2]*b[:,7]
    c3 = a[:,0]*b[:,2] + a[:,1]*b[:,5] + a[:,2]*b[:,8]

    c4 = a[:,3]*b[:,0] + a[:,4]*b[:,3] + a[:,5]*b[:,6]
    c5 = a[:,3]*b[:,1] + a[:,4]*b[:,4] + a[:,5]*b[:,7]
    c6 = a[:,3]*b[:,2] + a[:,4]*b[:,5] + a[:,5]*b[:,8]

    c7 = a[:,6]*b[:,0] + a[:,7]*b[:,3] + a[:,8]*b[:,6]
    c8 = a[:,6]*b[:,1] + a[:,7]*b[:,4] + a[:,8]*b[:,7]
    c9 = a[:,6]*b[:,2] + a[:,7]*b[:,5] + a[:,8]*b[:,8]

    Out = np.stack([c1, c2, c3, c4, c5, c6, c7, c8, c9], axis=1)

    return Out
