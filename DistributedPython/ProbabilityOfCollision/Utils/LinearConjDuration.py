import numpy as np
from .conj_bounds_Coppola import conj_bounds_Coppola
from .FindNearbyCA import FindNearbyCA

def LinearConjDuration(r1, v1, cov1, r2, v2, cov2, HBR, params=None):
    """
    LinearConjDuration - Calculate the time bounds, midpoint, duration, and
                         short-term encounter validity interval for a
                         linearized conjunction.

    Args:
        r1 (np.ndarray): Primary object's position vector [3x1] or [3,]
        v1 (np.ndarray): Primary object's velocity vector [3x1] or [3,]
        cov1 (np.ndarray): Primary object's state covariance matrix [3x3] or [6x6]
        r2 (np.ndarray): Secondary object's position vector [3x1] or [3,]
        v2 (np.ndarray): Secondary object's velocity vector [3x1] or [3,]
        cov2 (np.ndarray): Secondary object's state covariance matrix [3x3] or [6x6]
        HBR (float): Hard body radius
        params (dict, optional): Run parameters.
            gamma (float): Precision factor (default = 1e-16)
            FindCA (bool): Flag to refine CA point (default = False)
            verbose (bool): Flag for verbose operation (default = False)

    Returns:
        tuple:
            tau0 (float): Initial time bound
            tau1 (float): Final time bound
            dtau (float): Conjunction duration
            taum (float): Midpoint time
            delt (float): STEVI half-width
    """
    if params is None:
        params = {}

    gamma = params.get('gamma', 1e-16)
    if isinstance(gamma, (list, tuple, np.ndarray)):
         if np.size(gamma) > 1:
             raise ValueError('Invalid gamma parameter')
         gamma = np.asarray(gamma).item()

    if gamma <= 0 or gamma >= 1:
        raise ValueError('Invalid gamma parameter')

    find_ca = params.get('FindCA', False)
    verbose = params.get('verbose', False)

    r1 = np.asarray(r1).flatten()
    v1 = np.asarray(v1).flatten()
    r2 = np.asarray(r2).flatten()
    v2 = np.asarray(v2).flatten()
    cov1 = np.asarray(cov1)
    cov2 = np.asarray(cov2)

    if find_ca:
        X1 = np.concatenate((r1, v1))
        X2 = np.concatenate((r2, v2))
        dTCA, X1CA, X2CA = FindNearbyCA(X1, X2)
        r1 = X1CA[0:3]
        v1 = X1CA[3:6]
        r2 = X2CA[0:3]
        v2 = X2CA[3:6]

    # Relative position, velocity and covariance
    r = r2 - r1
    v = v2 - v1
    c = cov1 + cov2

    # Call the Coppola conjunction bounds function
    # conj_bounds_Coppola returns (tau0, tau1, tau0_gam1, tau1_gam1)
    tau0_arr, tau1_arr, _, _ = conj_bounds_Coppola(gamma, HBR, r, v, c, verbose)

    # Since gamma is scalar, tau0_arr is scalar (0-d array or float)
    if np.ndim(tau0_arr) == 0:
        tau0 = float(tau0_arr)
        tau1 = float(tau1_arr)
    else:
        tau0 = float(tau0_arr.item())
        tau1 = float(tau1_arr.item())

    # Calculate the conjunction duration, midpoint and STEVI
    dtau = tau1 - tau0
    taum = (tau1 + tau0) / 2
    delt = max(dtau, abs(tau0), abs(tau1))

    return tau0, tau1, dtau, taum, delt
