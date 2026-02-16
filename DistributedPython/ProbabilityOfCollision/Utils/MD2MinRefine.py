import numpy as np
from DistributedPython.ProbabilityOfCollision.Utils.PeakOverlapMD2 import PeakOverlapMD2

def MD2MinRefine(tmd, tsg, dtsg, ttol, itermax, findmin, Eb10, Qb10, Eb20, Qb20, HBR, POPPAR):
    """
    MD2MinRefine - Iteratively refine the parameters of the min of effective MD2 curve.

    Args:
        tmd (float): Time, in seconds from TCA, of the minimum of the Mahalanobis distance squared parabola.
        tsg (float): The 1-sigma time width for numerically computing the output values.
        dtsg (float): Sigma multiplier applied to the tsg parameter.
        ttol (float): Convergence tolerance (in seconds).
        itermax (int): Maximum number of iterations performed by the function.
        findmin (bool): When set to true, indicates that the function should search for the minimum.
        Eb10 (array-like): Equinoctial state of the primary object at TCA. [6x1].
        Qb10 (array-like): Equinoctial state covariance of the primary object at TCA. [6x6].
        Eb20 (array-like): Equinoctial state of the secondary object at TCA. [6x1].
        Qb20 (array-like): Equinoctial state covariance of the secondary object at TCA. [6x6].
        HBR (float): Combined HBR of the two objects (in m).
        POPPAR (dict): Peak Overlap (PeakOverlapMD2) input parameters.

    Returns:
        dict: Structure containing results and other pertinent parameters calculated by the function.
    """

    # Flatten inputs
    Eb10 = np.asarray(Eb10).flatten()
    Eb20 = np.asarray(Eb20).flatten()
    Qb10 = np.asarray(Qb10)
    Qb20 = np.asarray(Qb20)

    # Iteratively refine minimum of the MD^2 curve in time
    iterating = True
    out = {}
    out['iter'] = 0

    itermin = min(2, itermax)
    itermed = min(4, itermax)

    out['MD2converged'] = False
    out['POPconverged'] = True
    out['tmd'] = tmd
    out['tsg'] = tsg
    out['ymd'] = np.nan

    # Initialize variables to avoid unbound local error if loop doesn't define them
    # (though loop runs at least once or logic handles it)
    aumd = {}

    while iterating:

        # Increment iteration counter
        out['iter'] += 1

        # Save old value of best MD^2 minimization time
        tmdold = out['tmd']
        tsgold = out['tsg']

        # Calculate 3 points to estimate numerical derivatives

        # Point spacing
        out['delt'] = out['tsg'] * dtsg
        twodelt = 2 * out['delt']
        delt2 = out['delt']**2

        # Low and high bracketing points
        out['tlo'] = out['tmd'] - out['delt']
        out['thi'] = out['tmd'] + out['delt']

        # Calculate y = MD^2 for the three points

        # Low point
        out['ylo'], out['Xulo'], out['Pslo'], out['Asdetlo'], out['Asinvlo'], _, aulo = \
            PeakOverlapMD2(out['tlo'], 0.0, Eb10, Qb10, 0.0, Eb20, Qb20, HBR, 1, POPPAR)

        # Mid point
        if out['iter'] == 1 or findmin:
            out['ymd'], out['Xu'], out['Ps'], out['Asdet'], out['Asinv'], _, aumd = \
                PeakOverlapMD2(out['tmd'], 0.0, Eb10, Qb10, 0.0, Eb20, Qb20, HBR, 1, POPPAR)

        # High point
        out['yhi'], out['Xuhi'], out['Pshi'], out['Asdethi'], out['Asinvhi'], _, auhi = \
            PeakOverlapMD2(out['thi'], 0.0, Eb10, Qb10, 0.0, Eb20, Qb20, HBR, 1, POPPAR)

        if np.isnan(out['ylo']) or np.isnan(out['ymd']) or np.isnan(out['yhi']):

            # POP search unconverged for one or more of the three points
            iterating = False
            out['POPconverged'] = False

        else:

            # Calculate new value of SigmaT
            out['ydot'] = (out['yhi'] - out['ylo']) / twodelt
            out['ydotdot'] = (out['yhi'] - 2 * out['ymd'] + out['ylo']) / delt2

            # Avoid division by zero or negative sqrt
            if out['ydotdot'] <= 0:
                out['tsg'] = np.inf
            else:
                out['tsg'] = np.sqrt(2.0 / out['ydotdot'])

            # Analyze convergence criteria
            if findmin:
                # In this case, estimate the new time that minimizes the
                # effective MD^2 curve before analyzing convergence
                if out['ydotdot'] != 0:
                     out['tmd'] = out['tmd'] - out['ydot'] / out['ydotdot']

                cnvg = abs(out['tmd'] - tmdold) < ttol * out['tsg'] if not np.isinf(out['tsg']) else False
            else:
                cnvg = abs(out['tsg'] - tsgold) < ttol * out['tsg'] if not np.isinf(out['tsg']) else False

            # Check for Maha. distance convergence
            if not cnvg:
                if findmin:
                    MD2check = out['iter'] >= itermed
                else:
                    MD2check = out['iter'] >= itermin

                if MD2check:
                    # MD2conv = 1 - [aulo.MD2actual auhi.MD2actual]/aumd.MD2actual;
                    # In python: 1 - np.array([aulo['MD2actual'], auhi['MD2actual']]) / aumd['MD2actual']

                    denom = aumd['MD2actual']
                    if denom == 0:
                        # If MD2 is zero, check absolute difference instead of relative
                        diff = np.array([aulo['MD2actual'], auhi['MD2actual']])
                        if np.max(np.abs(diff)) < ttol:
                            cnvg = True
                    else:
                        MD2conv = 1.0 - np.array([aulo['MD2actual'], auhi['MD2actual']]) / denom
                        if np.max(np.abs(MD2conv)) < ttol:
                            cnvg = True

            if np.isinf(out['tsg']):
                # Second derivative not positive (i.e., ydotdot <= 0)
                iterating = False
                out['MD2converged'] = False
            elif cnvg and out['iter'] >= itermin:
                iterating = False
                out['MD2converged'] = True
            else:
                iterating = out['iter'] <= itermax

    return out
