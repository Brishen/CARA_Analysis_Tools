import numpy as np
import warnings
from DistributedPython.ProbabilityOfCollision.PcCircle import PcCircle
from DistributedPython.ProbabilityOfCollision.Utils.Product3x3 import Product3x3

def PcCircleWithConjData(r1, v1, cov1, r2, v2, cov2, HBR, params=None):
    """
    PcCircleWithConjData - Computes a Pc using the PcCircle method and adds
                           extra conjunction data used by CARA.

    Args:
        r1 (np.ndarray): Primary object's position vector in ECI coordinates [nx3]
        v1 (np.ndarray): Primary object's velocity vector in ECI coordinates [nx3 or 1x3]
        cov1 (np.ndarray): Primary object's covariance matrix [nx9, nx3x3, or nx6x6]
        r2 (np.ndarray): Secondary object's position vector in ECI coordinates [nx3]
        v2 (np.ndarray): Secondary object's velocity vector in ECI coordinates [nx3 or 1x3]
        cov2 (np.ndarray): Secondary object's covariance matrix [nx9, nx3x3, or nx6x6]
        HBR (float or np.ndarray): Hard body radius [1x1 or nx1]
        params (dict, optional): Auxiliary input parameter structure.

    Returns:
        tuple:
            Pc (np.ndarray): Probability of collision
            out (dict): An auxiliary output structure which contains a number of extra
                        quantities from the Pc calculation.
    """
    if params is None:
        params = {}

    # Call PcCircle
    Pc, out = PcCircle(r1, v1, cov1, r2, v2, cov2, HBR, params)

    # Get adjusted inputs from out
    r1 = out['r1']
    v1 = out['v1']
    cov1 = out['cov1']
    r2 = out['r2']
    v2 = out['v2']
    cov2 = out['cov2']

    # Miss Distance
    r = r1 - r2
    out['MissDistance'] = np.linalg.norm(r, axis=1)

    # Semimajor/semiminor axis
    out['SemiMajorAxis'] = out['sx']
    out['SemiMinorAxis'] = out['sz']

    # Clock Angle
    # PcCircle outputs 'EigV1' (eigenvector for sx)
    V1 = out['EigV1']
    ClockAngle = np.arctan2(V1[:, 1], V1[:, 0]) * 180.0 / np.pi

    adjDown = ClockAngle >= 90.0
    adjUp = ClockAngle <= -90.0
    ClockAngle[adjDown] -= 180.0
    ClockAngle[adjUp] += 180.0

    out['ClockAngle'] = ClockAngle

    # x1 Sigma
    sx = out['sx']
    sz = out['sz']

    CA_rad = np.deg2rad(ClockAngle)
    cosCA = np.cos(CA_rad)
    sinCA = np.sin(CA_rad)

    denom_sq = (sz * cosCA)**2 + (sx * sinCA)**2
    out['x1Sigma'] = (sx * sz) / np.sqrt(denom_sq)

    # Compute uncertainties in the primary RIC frame
    CombCov = cov1 + cov2

    # h1 = cross(r1, v1)
    h1 = np.cross(r1, v1)

    # Rhat = r1 / norm(r1)
    r1norm = np.linalg.norm(r1, axis=1, keepdims=True)
    Rhat = r1 / r1norm

    # Chat = h1 / norm(h1)
    h1norm = np.linalg.norm(h1, axis=1, keepdims=True)
    Chat = h1 / h1norm

    # Ihat = cross(Chat, Rhat)
    Ihat = np.cross(Chat, Rhat)

    # eci2ricPrim transformation matrix
    # Rows are Rhat, Ihat, Chat
    # Flattened row-major: Rhat components, Ihat components, Chat components
    eci2ricPrim = np.hstack([Rhat, Ihat, Chat])

    # M * C * M^T
    perm_indices = [0, 3, 6, 1, 4, 7, 2, 5, 8]
    eci2ricPrim_T = eci2ricPrim[:, perm_indices]

    CovCombRICPrim = Product3x3(eci2ricPrim, Product3x3(CombCov, eci2ricPrim_T))

    # RadialSigma = sqrt(CovCombRICPrim(1,1)) * 1000
    # In-trackSigma = sqrt(CovCombRICPrim(2,2)) * 1000
    # CrossTrackSigma = sqrt(CovCombRICPrim(3,3)) * 1000
    # Indices 0, 4, 8

    out['RadialSigma'] = np.sqrt(CovCombRICPrim[:, 0]) * 1000.0
    out['InTrackSigma'] = np.sqrt(CovCombRICPrim[:, 4]) * 1000.0
    out['CrossTrackSigma'] = np.sqrt(CovCombRICPrim[:, 8]) * 1000.0

    # Relative Phase Angle
    v1dotv2 = np.sum(v1 * v2, axis=1)
    norm_v1 = np.linalg.norm(v1, axis=1)
    norm_v2 = np.linalg.norm(v2, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
         cos_theta = v1dotv2 / (norm_v1 * norm_v2)
         # Clamp to [-1, 1] to avoid numerical errors
         cos_theta = np.clip(cos_theta, -1.0, 1.0)
         out['RelativePhaseAngle'] = np.arccos(cos_theta) * 180.0 / np.pi

    # Condition Numbers
    # Reshape covariances to (N, 3, 3)
    C1 = cov1.reshape(-1, 3, 3)
    C2 = cov2.reshape(-1, 3, 3)
    Ccomb = C1 + C2

    # Helper to safe cond
    def safe_cond(mats):
        try:
            return np.linalg.cond(mats)
        except Exception:
            # Fallback for older numpy or errors
            res = np.empty(mats.shape[0])
            for i in range(mats.shape[0]):
                try:
                    res[i] = np.linalg.cond(mats[i])
                except Exception:
                    res[i] = np.nan
            return res

    out['CondNumPrimary'] = safe_cond(C1)
    out['CondNumSecondary'] = safe_cond(C2)
    out['CondNumCombined'] = safe_cond(Ccomb)

    # Projected Covariance condition number
    Amat = out['Amat'] # (N, 3) -> [xx, xz, zz]
    Cp = np.empty((Amat.shape[0], 2, 2))
    Cp[:, 0, 0] = Amat[:, 0]
    Cp[:, 0, 1] = Amat[:, 1]
    Cp[:, 1, 0] = Amat[:, 1]
    Cp[:, 1, 1] = Amat[:, 2]

    out['CondNumProjected'] = safe_cond(Cp)

    return Pc, out
