import numpy as np
from scipy.integrate import dblquad

from DistributedPython.ProbabilityOfCollision.Utils.CovRemEigValClip import CovRemEigValClip

def Pc2D_Foster(r1, v1, cov1, r2, v2, cov2, HBR, RelTol, HBRType):
    """
    Computes 2D Pc according to Foster method. This function supports three
    different types of hard body regions: 'circle', 'square', and square
    equivalent to the area of the circle ('squareEquArea'). It also handles
    both 3x3 and 6x6 covariances but, by definition, the 2D Pc calculation
    only uses the 3x3 position covariance.

    Args:
        r1 (np.ndarray): Primary object's position vector in ECI coordinates (km) [1x3]
        v1 (np.ndarray): Primary object's velocity vector in ECI coordinates (km/s) [1x3]
        cov1 (np.ndarray): Primary object's covariance matrix in ECI coordinate frame [3x3] or [6x6]
        r2 (np.ndarray): Secondary object's position vector in ECI coordinates (km) [1x3]
        v2 (np.ndarray): Secondary object's velocity vector in ECI coordinates (km/s) [1x3]
        cov2 (np.ndarray): Secondary object's covariance matrix in ECI coordinate frame [3x3] or [6x6]
        HBR (float): Hard body region radius (km)
        RelTol (float): Tolerance used for double integration convergence
                        (usually set to the value of 1e-08)
        HBRType (str): Type of hard body region. (Must be one of the
                       following: 'circle', 'square', 'squareEquArea')

    Returns:
        tuple:
            Pc (float): Probability of collision
            Arem (np.ndarray): Combined covariance projected onto xz-plane in the
                               relative encounter frame. Also called Cp.
            IsPosDef (bool): Flag indicating if the combined, marginalized and
                             remediated covariance has a negative eigenvalue. If
                             the test failed the Pc is not computed. The
                             function returns NaN for Pc and an empty matrix for
                             ConjData. (Success = 1 & Fail = 0)
            IsRemediated (bool): Flag indicating if the combined and marginalized
                                 covariance was remediated
    """
    # Ensure inputs are numpy arrays
    r1 = np.asarray(r1).flatten()
    v1 = np.asarray(v1).flatten()
    cov1 = np.asarray(cov1)
    r2 = np.asarray(r2).flatten()
    v2 = np.asarray(v2).flatten()
    cov2 = np.asarray(cov2)

    # Combined position covariance
    covcomb = cov1[0:3, 0:3] + cov2[0:3, 0:3]

    # Construct relative encounter frame
    r = r1 - r2
    v = v1 - v2
    h = np.cross(r, v)

    # Relative encounter frame
    norm_v = np.linalg.norm(v)
    norm_h = np.linalg.norm(h)

    # Handle degenerate case if necessary, though not explicitly handled in MATLAB
    # Ideally should raise error or handle gracefully
    if norm_v == 0:
        raise ValueError("Relative velocity is zero, cannot define encounter frame.")
    if norm_h == 0:
        raise ValueError("Relative position and velocity are parallel, cannot define encounter frame.")

    y = v / norm_v
    z = h / norm_h
    x = np.cross(y, z)

    # Transformation matrix from ECI to relative encounter plane
    eci2xyz = np.vstack([x, y, z])

    # Transform combined ECI covariance into xyz
    covcombxyz = eci2xyz @ covcomb @ eci2xyz.T

    # Projection onto xz-plane in the relative encounter frame
    # Cp = [1 0 0; 0 0 1] * covcombxyz * [1 0 0; 0 0 1]'
    # This is effectively selecting rows 0 and 2 and columns 0 and 2
    # Indices 0 corresponds to x, 2 corresponds to z (since y is at index 1 in Python 0-based indexing)
    # Wait, eci2xyz = [x; y; z]. So row 0 is x, row 1 is y, row 2 is z.
    # Matlab: Cp = [1 0 0; 0 0 1] * covcombxyz * [1 0 0; 0 0 1]';
    # This selects row 1 and 3 (1-based) -> row 0 and 2 (0-based).
    # Correct.
    Cp = covcombxyz[[0, 2]][:, [0, 2]]

    # Remediate non-positive definite covariances
    Lclip = (1e-4 * HBR)**2

    rem_res = CovRemEigValClip(Cp, Lclip)
    Lrem = rem_res['Lrem']
    IsRemediated = rem_res['ClipStatus']
    Adet = rem_res['Adet']
    Ainv = rem_res['Ainv']
    Arem = rem_res['Arem']

    IsPosDef = np.min(Lrem) > 0

    if not IsPosDef:
        # In Python we raise an error similar to MATLAB
        raise ValueError('Combined position covariance matrix is not positive definite when mapped to the 2-D conjunction plane. Please review input data quality.')

    # CALCULATE DOUBLE INTEGRAL

    # Center of HBR in the relative encounter plane
    x0 = np.linalg.norm(r)
    z0 = 0.0

    # Inverse of the Cp matrix
    C = Ainv

    # Absolute Tolerance
    AbsTol = 1e-13

    # Integrand
    # dblquad expects func(y, x) where y is inner, x is outer.
    # Here inner integral is over z (from lower to upper semicircle), outer over x.
    # So map z -> y (inner variable), x -> x (outer variable).

    def integrand(z_val, x_val):
        # C is 2x2 matrix (indices 0,0 0,1 1,0 1,1)
        # Using z_val for z, x_val for x.
        term = -0.5 * (C[0, 0] * x_val**2 + (C[0, 1] + C[1, 0]) * z_val * x_val + C[1, 1] * z_val**2)
        return np.exp(term)

    # Depending on the type of hard body region, compute Pc
    HBRType_lower = HBRType.lower()

    if HBRType_lower == 'circle':

        def upper_semicircle(x_val):
            val = HBR**2 - (x_val - x0)**2
            if val < 0: return 0.0
            return np.sqrt(val)

        def lower_semicircle(x_val):
            val = HBR**2 - (x_val - x0)**2
            if val < 0: return 0.0
            return -np.sqrt(val)

        val, error_est = dblquad(integrand, x0 - HBR, x0 + HBR, lower_semicircle, upper_semicircle, epsabs=AbsTol, epsrel=RelTol)
        Pc = (1 / (2 * np.pi)) * (1 / np.sqrt(Adet)) * val

    elif HBRType_lower == 'square':
        val, error_est = dblquad(integrand, x0 - HBR, x0 + HBR, lambda x: z0 - HBR, lambda x: z0 + HBR, epsabs=AbsTol, epsrel=RelTol)
        Pc = (1 / (2 * np.pi)) * (1 / np.sqrt(Adet)) * val

    elif HBRType_lower == 'squareequarea':
        limit = (np.sqrt(np.pi) / 2) * HBR
        val, error_est = dblquad(integrand, x0 - limit, x0 + limit, lambda x: z0 - limit, lambda x: z0 + limit, epsabs=AbsTol, epsrel=RelTol)
        Pc = (1 / (2 * np.pi)) * (1 / np.sqrt(Adet)) * val

    else:
        raise ValueError(f"{HBRType} as HBRType is not supported...")

    return Pc, Arem, IsPosDef, IsRemediated
