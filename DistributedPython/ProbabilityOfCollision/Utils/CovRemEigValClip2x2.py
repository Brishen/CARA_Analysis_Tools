import numpy as np
from .eig2x2 import eig2x2

def CovRemEigValClip2x2(Araw, Lclip=None):
    """
    CovRemEigValClip2x2 - Vectorized covariance remediation using eigenvalue clipping for 2x2 matrices.

    Args:
        Araw (np.ndarray): Input raw 2x2 covariance matrices [nx3].
                           Araw[:,0] is (1,1)
                           Araw[:,1] is (1,2) and (2,1)
                           Araw[:,2] is (2,2)
        Lclip (float or np.ndarray, optional): Clipping limit for eigenvalues. Defaults to 0.
                                               Can be a scalar or an array of shape (n,) or (n,1).

    Returns:
        tuple:
            ClipStatus (np.ndarray): Boolean array [nx1] (or [n]) indicating if clipping occurred.
            Arem (np.ndarray): Remediated covariance matrices [nx3].
    """

    Araw = np.asarray(Araw, dtype=float)

    # Ensure Araw is 2D
    if Araw.ndim == 1:
        # If input is 1D (representing one matrix), reshape it to (1, 3)
        if Araw.size == 3:
            Araw = Araw.reshape(1, 3)
        else:
             raise ValueError('Array needs to be an nx3 matrix')

    szC = Araw.shape
    if szC[1] != 3:
        raise ValueError('Matrix needs to be an nx3 matrix to represent 2x2 symmetric covariances')

    num_matrices = szC[0]

    if Lclip is None:
        Lclip = 0.0

    Lclip = np.asarray(Lclip)

    if Lclip.ndim == 0:
        # Scalar
        Lclip = np.full(num_matrices, Lclip)
    else:
        # Array
        Lclip = Lclip.reshape(-1) # Flatten to 1D
        if Lclip.size != num_matrices:
             raise ValueError(f'Lclip size {Lclip.size} does not match number of matrices {num_matrices}')

    if not np.all(np.isreal(Lclip)):
        raise ValueError('Lclip must be real')
    if np.any(Lclip < 0):
        raise ValueError('Lclip cannot be negative')

    if not np.isrealobj(Araw):
        raise ValueError('Covariance matrix cannot have imaginary elements')

    # Calculate eigen-decomposition
    # eig2x2 returns V1, V2, L1, L2
    V1, V2, L1, L2 = eig2x2(Araw)

    # Lraw = [l1 l2]
    # L1, L2 are (N,) arrays (from my eig2x2 implementation)
    Lraw = np.column_stack((L1, L2))

    # Ensure eigenvalues and eigenvectors are real
    if np.iscomplexobj(V1) or np.iscomplexobj(V2) or np.iscomplexobj(L1) or np.iscomplexobj(L2):
         raise ValueError('Eigenvalues and eigenvectors must be real')

    # Clip eigenvalues
    Lrem = Lraw.copy()

    # Check if min eigenvalue < Lclip
    min_eig = np.min(Lraw, axis=1)
    ClipStatus = min_eig < Lclip

    # Remediate
    # Lrem(Lraw[:,1] < Lclip, 1) = Lclip
    # Note: Lraw[:, 0] is L1 (largest), Lraw[:, 1] is L2 (smallest)

    mask1 = Lraw[:, 0] < Lclip
    Lrem[mask1, 0] = Lclip[mask1]

    mask2 = Lraw[:, 1] < Lclip
    Lrem[mask2, 1] = Lclip[mask2]

    # Remediated covariance
    Arem = Araw.copy()
    idx = ClipStatus # boolean mask

    if np.any(idx):
        # In my python code:
        # V1 corresponds to L1 (col 0 of Lrem)
        # V2 corresponds to L2 (col 1 of Lrem)

        # V1 is [nx2]. V1[:,0] is x component, V1[:,1] is y component.

        # Formula:
        # term1 (1,1): v1_x^2 * l1 + v2_x^2 * l2
        # term2 (1,2): v1_x*v1_y * l1 + v2_x*v2_y * l2
        # term3 (2,2): v1_y^2 * l1 + v2_y^2 * l2

        v1_x = V1[idx, 0]
        v1_y = V1[idx, 1]
        v2_x = V2[idx, 0]
        v2_y = V2[idx, 1]

        l1_rem = Lrem[idx, 0]
        l2_rem = Lrem[idx, 1]

        c11 = v1_x**2 * l1_rem + v2_x**2 * l2_rem
        c12 = v1_x * v1_y * l1_rem + v2_x * v2_y * l2_rem
        c22 = v1_y**2 * l1_rem + v2_y**2 * l2_rem

        Arem[idx, 0] = c11
        Arem[idx, 1] = c12
        Arem[idx, 2] = c22

    return ClipStatus, Arem
