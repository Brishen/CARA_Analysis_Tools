import numpy as np
import warnings
from scipy.integrate import quad
from DistributedPython.ProbabilityOfCollision.Utils.CheckAndResizePosVel import CheckAndResizePosVel
from DistributedPython.ProbabilityOfCollision.Utils.CheckAndResizeCov import CheckAndResizeCov
from DistributedPython.ProbabilityOfCollision.Utils.Product3x3 import Product3x3
from DistributedPython.ProbabilityOfCollision.Utils.eig2x2 import eig2x2
from DistributedPython.Utils.AugmentedMath.erf_vec_dif import erf_vec_dif
from DistributedPython.Utils.AugmentedMath.GenGCQuad import GenGCQuad

def Pc2DIntegrand(x, xm, zm, dx, dz, R2):
    """
    Vectorized integrand for the 1D integral along x-axis of the conjunction plane.
    """
    # Ensure x is array-like for vectorized operations
    x_arr = np.atleast_1d(x)

    val_sq = R2 - (x_arr - xm)**2
    # Ensure non-negative before sqrt
    Rx = np.sqrt(np.maximum(0, val_sq))

    arg1 = (zm + Rx) / dz
    arg2 = (zm - Rx) / dz

    term1 = np.exp(-(x_arr / dx)**2)
    term2 = erf_vec_dif(arg1, arg2)

    result = term1 * term2

    # Return scalar if input was scalar (required for scipy.integrate.quad)
    if np.isscalar(x):
        return result.item()
    return result

def PcCircle(r1, v1, cov1, r2, v2, cov2, HBR, params=None):
    """
    PcCircle - Computes Pc for vectorized state/cov input by integrating over
    a circle on the conjunction plane.

    Args:
        r1 (np.ndarray): Primary object's position vector in ECI coordinates [nx3]
        v1 (np.ndarray): Primary object's velocity vector in ECI coordinates [nx3 or 1x3]
        cov1 (np.ndarray): Primary object's covariance matrix [nx9, nx3x3, or nx6x6]
        r2 (np.ndarray): Secondary object's position vector in ECI coordinates [nx3]
        v2 (np.ndarray): Secondary object's velocity vector in ECI coordinates [nx3 or 1x3]
        cov2 (np.ndarray): Secondary object's covariance matrix [nx9, nx3x3, or nx6x6]
        HBR (float or np.ndarray): Hard body radius [1x1 or nx1]
        params (dict, optional): Auxiliary input parameter structure. Defaults to None.
            EstimationMode (int): Default 64.
            WarningLevel (int): Default 0.
            PriSecCovProcessing (bool): Default False.

    Returns:
        tuple:
            Pc (np.ndarray): Probability of collision [nx1]
            out (dict): Auxiliary output dictionary
    """
    if params is None:
        params = {}

    EstimationMode = params.get('EstimationMode', 64)
    WarningLevel = params.get('WarningLevel', 0)
    PriSecCovProcessing = params.get('PriSecCovProcessing', False)

    # Check for valid EstimationMode
    if EstimationMode <= 0:
        if EstimationMode != 0 and EstimationMode != -1:
            raise ValueError('PcCircle:InvalidEstimationMode: Invalid EstimationMode')
    else:
        if EstimationMode != round(EstimationMode):
            raise ValueError('PcCircle:InvalidEstimationMode: Invalid EstimationMode')
        elif EstimationMode < 16 and WarningLevel > 0:
            warnings.warn('PcCircle:InsufficientEstimationMode: EstimationMode specifies fewer than 16 quadrature points, which can cause inaccurate Pc estimates')

    # Reformat the inputs to expected dimensions
    Nvec, v1 = CheckAndResizePosVel(r1, v1)
    Nvec2, v2 = CheckAndResizePosVel(r2, v2)

    if Nvec != Nvec2:
        raise ValueError('PcCircle:UnequalPositionCount: Number of primary and secondary positions must be equal')

    cov1 = CheckAndResizeCov(Nvec, cov1)
    cov2 = CheckAndResizeCov(Nvec2, cov2)

    # Replicate scalar HBR into an nx1 array
    HBR = np.asarray(HBR)
    if HBR.size == 1:
        if Nvec > 1:
            HBR = np.full((Nvec, 1), HBR.item())
        else:
            HBR = HBR.reshape(1, 1)
    elif HBR.shape != (Nvec, 1):
        if HBR.ndim == 1 and HBR.shape[0] == Nvec:
             HBR = HBR.reshape(Nvec, 1)
        else:
             raise ValueError('PcCircle:InvalidHBRDimensions: Input HBR array dimension must be 1x1 or nx1')

    # Ensure HBR values are nonnegative
    if np.any(HBR < 0):
        if WarningLevel > 0:
            warnings.warn('PcCircle:NegativeHBR: Negative HBR values found and replaced with zeros')
        HBR[HBR < 0] = 0

    # Save the adjusted input parameters into the output structure
    out = {
        'r1': r1, 'v1': v1, 'cov1': cov1,
        'r2': r2, 'v2': v2, 'cov2': cov2,
        'HBR': HBR
    }

    # Combine the covariances
    # cov1 and cov2 are nx9 flattened row-major
    CombCov = cov1 + cov2

    # Relative position and velocity
    r = r1 - r2
    v = v1 - v2

    # Check and adjust for zero miss distance
    rmag = np.linalg.norm(r, axis=1, keepdims=True)

    # eps(rmag) in MATLAB is distance from abs(X) to the next larger floating point number.
    # In numpy, np.spacing(X)
    reps = np.maximum(10.0 * np.spacing(rmag), 1e-6 * HBR)

    SmallRmag = (rmag < reps).flatten()

    if np.any(SmallRmag):
        if WarningLevel > 0:
             warnings.warn('PcCircle:ZeroMissDistance: Zero or near-zero miss distance cases found; perturbing miss distance for those cases')

        idx = SmallRmag
        rsum = r1[idx] + r2[idx]
        rsummag = np.linalg.norm(rsum, axis=1, keepdims=True)
        vmag_small = np.linalg.norm(v[idx], axis=1, keepdims=True)

        rdel = reps[idx] * np.cross(rsum, v[idx]) / rsummag / vmag_small
        r[idx] = r[idx] + rdel

    # Check for zero relative velocity
    vmag = np.linalg.norm(v, axis=1, keepdims=True)
    ZeroVmag = (vmag == 0).flatten()

    if np.any(ZeroVmag) and WarningLevel > 0:
        warnings.warn('PcCircle:ZeroRelativeVelocity: Zero relative velocity cases found; setting Pc to NaN for those cases')

    # Orbit normal
    h = np.cross(r, v)
    hmag = np.linalg.norm(h, axis=1, keepdims=True)

    with np.errstate(divide='ignore', invalid='ignore'):
        y = v / vmag
        z = h / hmag

    x = np.cross(y, z)

    eci2xyz = np.hstack([x, y, z])

    out['xhat'] = x
    out['yhat'] = y
    out['zhat'] = z

    # Pri and Sec cov processing
    if PriSecCovProcessing:
        perm_indices = [0, 3, 6, 1, 4, 7, 2, 5, 8]
        eci2xyz_T = eci2xyz[:, perm_indices]

        RotatedCov = Product3x3(eci2xyz, Product3x3(cov1, eci2xyz_T))
        AmatPri = np.column_stack([RotatedCov[:, 0], RotatedCov[:, 2], RotatedCov[:, 8]])
        out['AmatPri'] = AmatPri

        V1, V2, L1, L2 = eig2x2(AmatPri)
        out['EigV1Pri'] = V1
        out['EigL1Pri'] = L1
        out['EigV2Pri'] = V2
        out['EigL2Pri'] = L2

        RotatedCov = Product3x3(eci2xyz, Product3x3(cov2, eci2xyz_T))
        AmatSec = np.column_stack([RotatedCov[:, 0], RotatedCov[:, 2], RotatedCov[:, 8]])
        out['AmatSec'] = AmatSec

        V1, V2, L1, L2 = eig2x2(AmatSec)
        out['EigV1Sec'] = V1
        out['EigL1Sec'] = L1
        out['EigV2Sec'] = V2
        out['EigL2Sec'] = L2

    # Project combined covariance
    perm_indices = [0, 3, 6, 1, 4, 7, 2, 5, 8]
    eci2xyz_T = eci2xyz[:, perm_indices]
    RotatedCov = Product3x3(eci2xyz, Product3x3(CombCov, eci2xyz_T))

    Amat = np.column_stack([RotatedCov[:, 0], RotatedCov[:, 2], RotatedCov[:, 8]])
    out['Amat'] = Amat

    V1, V2, L1, L2 = eig2x2(Amat)
    out['EigV1'] = V1
    out['EigL1'] = L1
    out['EigV2'] = V2
    out['EigL2'] = L2

    # Issue error if any cases are found with two non-positive eigenvalues
    if np.any(L1 <= 0):
        raise ValueError('PcCircle:TwoNonPositiveEigenvalues: Invalid case(s) found with two non-positive eigenvalues')

    # Issue a warning for any NPD cases
    if WarningLevel > 0 and np.any(L2 <= 0):
        warnings.warn('PcCircle:NPDCovariance: NPD covariance(s) found; remediating using eigenvalue clipping method')

    # Eigenvalue clipping
    FiniteHBR = ~np.isinf(HBR).flatten()
    Fclip = 1e-4
    Lrem = (Fclip * HBR.flatten())**2

    IsRem1 = (L1 < Lrem) & FiniteHBR
    L1[IsRem1] = Lrem[IsRem1]

    IsRem2 = (L2 < Lrem) & FiniteHBR
    L2[IsRem2] = Lrem[IsRem2]

    out['IsPosDef'] = L2 > 0
    out['IsRemediated'] = IsRem1 | IsRem2

    sx = np.sqrt(L1)
    sz = np.sqrt(L2)
    out['sx'] = sx
    out['sz'] = sz

    # The miss distance coordinates in the conjunction plane (xm, zm)
    rm = np.linalg.norm(r, axis=1)
    xm = rm * np.abs(V1[:, 0])
    zm = rm * np.abs(V1[:, 1])

    out['xm'] = xm
    out['zm'] = zm

    # Estimate the Pc
    Pc = np.full(Nvec, np.nan)

    if EstimationMode <= 0:
        if EstimationMode == 0:
             # Equal-area square approximation
             HSQ = np.sqrt(np.pi / 4.0) * HBR.flatten()
        else:
             # Circumscribing square upper bound
             HSQ = HBR.flatten()

        sqrt2 = np.sqrt(2.0)
        dx = sqrt2 * sx
        dz = sqrt2 * sz

        Ex = erf_vec_dif( (xm + HSQ)/dx, (xm - HSQ)/dx )
        Ez = erf_vec_dif( (zm + HSQ)/dz, (zm - HSQ)/dz )

        Pc = Ex * Ez / 4.0

    else:
        # Integration
        xlo = xm - HBR.flatten()
        xhi = xm + HBR.flatten()

        Nsx = 4.0 * sx

        # xloclip = xlo; xloclip(xlo < 0) = 0;
        xloclip = np.maximum(xlo, 0)

        # out.ClipBoundSet = ~( (xlo > -Nsx) & (xhi < xloclip+Nsx) );
        ClipBoundSet = ~((xlo > -Nsx) & (xhi < xloclip + Nsx))
        out['ClipBoundSet'] = ClipBoundSet

        if EstimationMode == 1:
            Iset = np.zeros(Nvec, dtype=bool)
        else:
            Iset = ~ClipBoundSet

        sqrt2 = np.sqrt(2.0)
        dx = sqrt2 * sx
        dz = sqrt2 * sz

        # GC Quadrature
        if np.any(Iset):
            # Calculate GC quadrature points and weights
            xGC, yGC, wGC = GenGCQuad(EstimationMode)

            # Subsets
            zm_sub = zm[Iset][:, np.newaxis]
            dz_sub = dz[Iset][:, np.newaxis]
            HBR_sub = HBR.flatten()[Iset][:, np.newaxis]
            xm_sub = xm[Iset][:, np.newaxis]
            dx_sub = dx[Iset][:, np.newaxis]
            sx_sub = sx[Iset][:, np.newaxis]

            # xrep = xm + HBR * xGC
            xrep = xm_sub + HBR_sub * xGC

            # Hxrep = HBR * yGC
            Hxrep = HBR_sub * yGC

            # Fint
            term1 = np.exp(-(xrep / dx_sub)**2)

            arg1 = (zm_sub + Hxrep) / dz_sub
            arg2 = (zm_sub - Hxrep) / dz_sub
            term2 = erf_vec_dif(arg1, arg2)

            Fint = term1 * term2

            # Psum = sum(wGC * Fint, axis=1)
            Psum = np.sum(wGC * Fint, axis=1)

            # Pc = (HBR/sx) * Psum
            Pc[Iset] = (HBR.flatten()[Iset] / sx[Iset]) * Psum

        # Fallback to quad
        Iset = (~Iset) & (~ZeroVmag)
        if np.any(Iset):
            indices = np.where(Iset)[0]

            # Adjust limits
            NegSet = Iset & (xlo < 0)
            if np.any(NegSet):
                 Nsx5 = 5.0 * sx
                 mNsx5 = -Nsx5

                 mask1 = NegSet & (xlo < mNsx5)
                 xlo[mask1] = mNsx5[mask1]

                 mask2 = NegSet & (xhi > Nsx5)
                 xhi[mask2] = Nsx5[mask2]

            HBR2 = HBR.flatten()**2

            for k in indices:
                func = lambda xx: Pc2DIntegrand(xx, xm[k], zm[k], dx[k], dz[k], HBR2[k])

                try:
                    res, err = quad(func, xlo[k], xhi[k], epsabs=1e-100, epsrel=1e-6, limit=100)
                except Exception:
                    res = np.nan

                Pc[k] = res

            # Final factor
            # Pc(Iset) = (Pc(Iset)./sx(Iset))/sqrt(8*pi);
            Pc[Iset] = (Pc[Iset] / sx[Iset]) / np.sqrt(8.0 * np.pi)

    # Post processing

    # Infinite HBR -> 1
    InfHBR = np.isinf(HBR.flatten())
    if np.any(InfHBR):
        Pc[InfHBR] = 1.0

    # Zero HBR -> 0
    ZeroHBR = (HBR.flatten() == 0)
    if np.any(ZeroHBR):
        Pc[ZeroHBR] = 0.0

    # Zero Relative Velocity -> NaN
    if np.any(ZeroVmag):
        Pc[ZeroVmag] = np.nan

    return Pc, out
