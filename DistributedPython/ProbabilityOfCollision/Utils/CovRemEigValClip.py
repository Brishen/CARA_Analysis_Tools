import numpy as np

def CovRemEigValClip(Araw, Lclip=0.0, Lraw=None, Vraw=None):
    """
    CovRemEigValClip - Remediate covariance matrix using eigenvalue clipping.

    This function detects non-positive definite (NPD) covariance matrices and
    performs adjustments to make them positive semi-definite (PSD) or positive
    definite (PD) using eigenvalue clipping.

    Args:
        Araw (np.ndarray): Input raw covariance matrix [NxN].
        Lclip (float, optional): Clipping limit for eigenvalues. Defaults to 0.0.
        Lraw (np.ndarray, optional): Eigenvalues of Araw [Nx1].
        Vraw (np.ndarray, optional): Eigenvector matrix of Araw [NxN].

    Returns:
        dict: containing:
            'Lrem': Remediated eigenvalues [Nx1].
            'Lraw': Unremediated eigenvalues [Nx1].
            'Vraw': Eigenvector matrix [NxN].
            'PosDefStatus': PD status (-1=NPD, 0=PSD, 1=PD).
            'ClipStatus': Boolean indicating if clipping occurred.
            'Adet': Determinant of remediated covariance.
            'Ainv': Inverse of remediated covariance.
            'Arem': Remediated covariance matrix.
    """

    # Input validation
    if Lclip is None:
        Lclip = 0.0

    if not isinstance(Lclip, (int, float, np.number)):
         raise ValueError('Lclip must be a number')

    if Lclip < 0:
        raise ValueError('Lclip cannot be negative')

    if (Lraw is None) != (Vraw is None):
        raise ValueError('Lraw or Vraw must both be input, or both not input')

    Araw = np.asarray(Araw, dtype=float)

    if Araw.ndim != 2:
        raise ValueError('Array needs to be a 2D matrix to represent a covariance')

    if Araw.shape[0] != Araw.shape[1]:
        raise ValueError('Matrix needs to be square to represent a covariance')

    if not np.all(np.isreal(Araw)):
        raise ValueError('Covariance matrix cannot have imaginary elements')

    # Eigen-decomposition
    if Lraw is None:
        # eigh returns eigenvalues in ascending order
        # For symmetric matrices, use eigh
        Lraw, Vraw = np.linalg.eigh(Araw)
    else:
        Lraw = np.asarray(Lraw, dtype=float)
        Vraw = np.asarray(Vraw, dtype=float)

    if not np.all(np.isreal(Lraw)) or not np.all(np.isreal(Vraw)):
        raise ValueError('Eigenvalues and eigenvectors must be real')

    # Ensure Lraw is 1D array
    Lraw = Lraw.flatten()

    # Positive Definite Status
    min_eig = np.min(Lraw)
    PosDefStatus = np.sign(min_eig)

    # Clipping
    Lrem = Lraw.copy()
    ClipStatus = False

    if min_eig < Lclip:
        ClipStatus = True
        Lrem[Lraw < Lclip] = Lclip

    # Calculate outputs
    Adet = np.prod(Lrem)

    # Ainv calculation: V * diag(1/L) * V'
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_Lrem = 1.0 / Lrem

    # Vraw * inv_Lrem scales the columns of Vraw (since inv_Lrem broadcasts to the last dimension)
    # Then we multiply by Vraw.T
    Ainv = (Vraw * inv_Lrem) @ Vraw.T

    if ClipStatus:
        # Reconstruct remediated covariance using remediated eigenvalues
        Arem = (Vraw * Lrem) @ Vraw.T
    else:
        Arem = Araw.copy()

    return {
        'Lrem': Lrem,
        'Lraw': Lraw,
        'Vraw': Vraw,
        'PosDefStatus': PosDefStatus,
        'ClipStatus': ClipStatus,
        'Adet': Adet,
        'Ainv': Ainv,
        'Arem': Arem
    }
