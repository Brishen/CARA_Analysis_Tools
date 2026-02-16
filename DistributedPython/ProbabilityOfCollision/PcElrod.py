import numpy as np
import warnings
from scipy.special import erfc
from DistributedPython.ProbabilityOfCollision.Utils.CheckAndResizePosVel import CheckAndResizePosVel
from DistributedPython.ProbabilityOfCollision.Utils.CheckAndResizeCov import CheckAndResizeCov
from DistributedPython.ProbabilityOfCollision.Utils.Product3x3 import Product3x3
from DistributedPython.ProbabilityOfCollision.Utils.RemediateCovariance2x2 import RemediateCovariance2x2
from DistributedPython.Utils.AugmentedMath.GenGCQuad import GenGCQuad

def PcElrod(r1, v1, cov1, r2, v2, cov2, HBR, ChebyshevOrder=64, WarningLevel=0):
    """
    PcElrod - Computes 2D Pc using the Chebyshev Gaussian Quadrature method
              (also known as the error function method).

    Args:
        r1 (np.ndarray): Primary object's position vector in ECI coordinates [nx3]
        v1 (np.ndarray): Primary object's velocity vector in ECI coordinates [nx3 or 1x3]
        cov1 (np.ndarray): Primary object's covariance matrix [nx9, nx3x3, or nx6x6]
        r2 (np.ndarray): Secondary object's position vector in ECI coordinates [nx3]
        v2 (np.ndarray): Secondary object's velocity vector in ECI coordinates [nx3 or 1x3]
        cov2 (np.ndarray): Secondary object's covariance matrix [nx9, nx3x3, or nx6x6]
        HBR (float or np.ndarray): Hard body radius [1x1 or nx1]
        ChebyshevOrder (int, optional): Order of Chebyshev polynomial. Defaults to 64.
        WarningLevel (int, optional): Warning level for NPD covariance remediation. Defaults to 0.

    Returns:
        tuple:
            Pc (np.ndarray): Probability of collision [nx1]
            Arem (np.ndarray): Combined covariance projected onto xz-plane [nx3]
            IsPosDef (np.ndarray): Flag indicating if PD [nx1]
            IsRemediated (np.ndarray): Flag indicating if remediated [nx1]
    """
    # Ensure ChebyshevOrder is even
    if ChebyshevOrder is None:
        ChebyshevOrder = 64
    else:
        ChebyshevOrder = int(np.ceil(ChebyshevOrder / 2) * 2)

    if WarningLevel is None:
        WarningLevel = 0

    # Reformat the inputs to expected dimensions
    numR1, v1 = CheckAndResizePosVel(r1, v1)
    numR2, v2 = CheckAndResizePosVel(r2, v2)

    if numR1 != numR2:
        raise ValueError('PcElrod:UnequalPositionCount: Number of primary and secondary positions must be equal')

    cov1 = CheckAndResizeCov(numR1, cov1)
    cov2 = CheckAndResizeCov(numR2, cov2)

    # Replicate scalar HBR into an nx1 array
    HBR = np.asarray(HBR, dtype=float)
    if HBR.ndim == 0:
        HBR = np.full((numR1, 1), HBR.item())
    elif HBR.size == 1:
        HBR = np.full((numR1, 1), HBR.item())
    else:
        HBR = HBR.reshape(-1, 1)
        if HBR.shape[0] != numR1:
             raise ValueError('PcElrod:InvalidHBRDimensions: Input HBR array dimension must be 1x1 or nx1')

    # Ensure HBR values are nonnegative
    if np.any(HBR < 0):
        if WarningLevel > 0:
            warnings.warn('PcElrod:NegativeHBR: Negative HBR values found and replaced with zeros')
        HBR[HBR < 0] = 0

    # Get the Chebyshev nodes and weights
    # GenGCQuad returns (1, N) arrays
    xGC, _, wGC = GenGCQuad(ChebyshevOrder)

    # Only retain the 2nd half of the nodes and weights
    # MATLAB: nGC = n(numGC/2+1:end);
    mid = ChebyshevOrder // 2
    nGC = xGC[0, mid:]
    wGC = wGC[0, mid:]

    # Combine the covariances
    CombCov = cov1 + cov2

    # Constructs relative encounter frame
    r = r1 - r2
    v = v1 - v2

    # Handle zero velocity cases by setting NaNs/handling appropriately
    # The original MATLAB code does: y=v./(sqrt(v(:,1).^2+v(:,2).^2+v(:,3).^2));
    # If v is 0, y is NaN.
    # We will use divide='ignore' to allow NaNs to propagate, which is likely intended or at least consistent.

    vmag = np.linalg.norm(v, axis=1, keepdims=True)
    h = np.cross(r, v)
    hmag = np.linalg.norm(h, axis=1, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        y = v / vmag
        z = h / hmag

    x = np.cross(y, z)
    eci2xyz = np.hstack([x, y, z])

    # Project combined covariances into conjunction planes
    # Transpose eci2xyz for rotation (assumed flattened row-major)
    # In MATLAB: eci2xyz(:,[1 4 7 2 5 8 3 6 9])
    perm_indices = [0, 3, 6, 1, 4, 7, 2, 5, 8]
    eci2xyz_T = eci2xyz[:, perm_indices]

    RotatedCov = Product3x3(eci2xyz, Product3x3(CombCov, eci2xyz_T))

    # ProjectedCov = RotatedCov(:,[1 3 9]) corresponds to indices 0, 2, 8 in row-major
    ProjectedCov = RotatedCov[:, [0, 2, 8]]

    # Remediate any non-positive definite covariances
    Arem, RevCholCov, IsPosDef, IsRemediated = RemediateCovariance2x2(ProjectedCov, HBR, WarningLevel)

    # Reverse Cholesky factorizations of projected covariances
    U = RevCholCov

    # other items needed for Gaussian quadrature integration
    # U is (N, 3). Columns [a, b, c] corresponding to MATLAB [1, 2, 3]
    Denominator = U[:, 0] * np.sqrt(2.0)

    # s = HBR./U(:,3);
    with np.errstate(divide='ignore', invalid='ignore'):
        s = HBR.flatten() / U[:, 2]

    HBR2 = HBR.flatten()**2

    # miss distances
    x0 = np.linalg.norm(r, axis=1)

    # error function calculation and summation for each Chebyshev node
    # Vectorized over nodes (k) and cases (N)

    # z = nGC(k).*s;
    # nGC is (K,). s is (N,). z_all will be (N, K).
    z_all = s[:, None] * nGC[None, :]

    # Radical = sqrt(HBR2-U(:,3).^2.*z.^2);
    # U(:,3) is U[:,2]. HBR2 is (N,).
    Radical_sq = HBR2[:, None] - (U[:, 2][:, None]**2) * (z_all**2)
    # Ensure non-negative due to floating point noise
    Radical = np.sqrt(np.maximum(0.0, Radical_sq))

    # Prepare broadcasting for Term calculations
    Denominator_col = Denominator[:, None]
    U2_col = U[:, 1][:, None]
    x0_col = x0[:, None]

    # Term1 = erfc((x0-U(:,2).*z-Radical)./Denominator);
    Term1 = erfc((x0_col - U2_col * z_all - Radical) / Denominator_col)

    # Term2 = erfc((x0+U(:,2).*z-Radical)./Denominator);
    Term2 = erfc((x0_col + U2_col * z_all - Radical) / Denominator_col)

    # Term3 = erfc((x0-U(:,2).*z+Radical)./Denominator);
    Term3 = erfc((x0_col - U2_col * z_all + Radical) / Denominator_col)

    # Term4 = erfc((x0+U(:,2).*z+Radical)./Denominator);
    Term4 = erfc((x0_col + U2_col * z_all + Radical) / Denominator_col)

    # Sum(:,k) = wGC(k).*exp(z.^2/-2).*(Term1+Term2-Term3-Term4);
    wGC_row = wGC[None, :]

    Sum = wGC_row * np.exp(z_all**2 / -2.0) * (Term1 + Term2 - Term3 - Term4)

    # Pc = sum(Sum,2).*s;
    Pc = np.sum(Sum, axis=1) * s

    return Pc, Arem, IsPosDef, IsRemediated
