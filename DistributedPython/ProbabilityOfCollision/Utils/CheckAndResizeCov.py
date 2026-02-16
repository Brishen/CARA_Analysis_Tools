import numpy as np

def CheckAndResizeCov(numR, cov):
    """
    CheckAndResizeCov - Resizes the covariance passed in into an n x 9 matrix.

    Format:
        cov = [C(1,1) C(1,2) C(1,3) C(2,1) C(2,2) C(2,3) C(3,1) C(3,2) C(3,3)] (Row major)

    Args:
        numR (int): Number of rows expected.
        cov (np.ndarray): Input covariance. Can be (n,9), (3,3), (6,6), (numR, 3, 3), (numR, 6, 6).

    Returns:
        np.ndarray: Resized covariance [numR, 9]
    """
    cov = np.asarray(cov)
    covSize = cov.shape

    # Python ndim check
    ndim = cov.ndim

    if ndim == 2:
        rows, cols = covSize

        if cols == 9:
             if rows == numR:
                 pass
             elif rows == 1:
                 cov = np.tile(cov, (numR, 1))
             else:
                 raise ValueError('2D Covariance cannot be resized to match r matrix')

        elif rows == 3 and cols == 3:
             # (3,3) matrix. Flatten row-major.
             cov = cov.reshape(1, 9)
             cov = np.tile(cov, (numR, 1))

        elif rows == 6 and cols == 6:
             cov = cov[0:3, 0:3]
             cov = cov.reshape(1, 9)
             cov = np.tile(cov, (numR, 1))

        else:
             raise ValueError('2D Covariance matrix must have 9 columns or be a 3x3 or 6x6 matrix!')

    elif ndim == 3:
        # Assuming (numR, rows, cols)

        if covSize[0] == numR:
            if covSize[1] == 3 and covSize[2] == 3:
                # (numR, 3, 3)
                cov = cov.reshape(numR, 9)
            elif covSize[1] == 6 and covSize[2] == 6:
                # (numR, 6, 6)
                cov = cov[:, 0:3, 0:3]
                cov = cov.reshape(numR, 9)
            else:
                raise ValueError('3D covariance matrix must be of size numRx3x3 or numRx6x6')
        else:
            raise ValueError('3D covariance matrix must be of size numRx3x3 or numRx6x6')

    else:
         raise ValueError('Improperly sized covariance was detected')

    return cov
