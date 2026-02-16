import numpy as np
from scipy.special import erfcinv

def conj_bounds_Coppola(gamma, HBR, rci, vci, Pci, verbose=False):
    """
    conj_bounds_Coppola - Calculate bounding times (tau0, tau1) for a linear conjunction event.

    Args:
        gamma (float or np.ndarray): Precision factor(s) for the calculation.
        HBR (float): Combined-object hard-body radius.
        rci (np.ndarray): Inertial relative position at TCA [3x1] or [3,].
        vci (np.ndarray): Inertial relative velocity at TCA [3x1] or [3,].
        Pci (np.ndarray): Inertial covariance matrix at TCA [3x3] or [6x6].
        verbose (bool): Verbosity flag.

    Returns:
        tuple:
            tau0 (np.ndarray): Initial bounding time(s)
            tau1 (np.ndarray): Final bounding time(s)
            tau0_gam1 (float): Initial bounding time for gamma=1
            tau1_gam1 (float): Final bounding time for gamma=1
    """
    if verbose:
        print(' ')
        print('Calculating conjunction time bounds using the Coppola (2012b) formulation:')

    gamma = np.asarray(gamma)
    Sgamma = gamma.shape

    vci = np.asarray(vci).flatten()
    rci = np.asarray(rci).flatten()
    Pci = np.asarray(Pci)

    v0mag = np.linalg.norm(vci)

    if v0mag < 100 * np.finfo(float).eps:
        # Initialize output time bounds
        if gamma.ndim == 0:
             tau0 = -np.inf
             tau1 = np.inf
        else:
             tau0 = np.full(Sgamma, -np.inf)
             tau1 = np.full(Sgamma, np.inf)

        tau0_gam1 = -np.inf
        tau1_gam1 = np.inf
        return tau0, tau1, tau0_gam1, tau1_gam1

    xhat = vci / v0mag

    # yhat = rci - xhat * (xhat' * rci)
    yhat = rci - xhat * np.dot(xhat, rci)
    yhat_norm = np.linalg.norm(yhat)

    if yhat_norm == 0:
        # Handle case where rci is parallel to vci
        if abs(xhat[0]) < 0.9:
            arb = np.array([1, 0, 0])
        else:
            arb = np.array([0, 1, 0])
        yhat = arb - xhat * np.dot(xhat, arb)
        yhat = yhat / np.linalg.norm(yhat)
    else:
        yhat = yhat / yhat_norm

    zhat = np.cross(xhat, yhat)

    # eROTi = [xhat'; yhat'; zhat']
    eROTi = np.vstack((xhat, yhat, zhat))

    # Extract 3x3 relative inertial position covariance
    # Aci = Pci(1:3,1:3)
    Aci = Pci[0:3, 0:3]

    # Rotate inputs into encounter frame
    rce = np.dot(eROTi, rci)
    Ace = np.dot(eROTi, np.dot(Aci, eROTi.T))

    # Extract quantities
    eta2 = Ace[0, 0]
    w = Ace[1:3, 0] # [2,1] and [3,1] -> indices 1 and 2 (0-based)
    Pc = Ace[1:3, 1:3] # [2:3, 2:3] -> indices 1:3, 1:3

    # b = inv(Pc) * w
    try:
        b = np.linalg.solve(Pc, w)
    except np.linalg.LinAlgError:
        b = np.linalg.lstsq(Pc, w, rcond=None)[0]

    # sv2 = sqrt(max(0, 2 * (eta2 - b'*w ) ))
    term = 2 * (eta2 - np.dot(b, w))
    sv2 = np.sqrt(max(0, term))

    q0 = np.dot(b, rce[1:3])
    bTb = np.dot(b, b)

    dmin = -HBR * np.sqrt(bTb)
    dmax = HBR * np.sqrt(1 + bTb)

    q0_minus_dmax = q0 - dmax
    q0_minus_dmin = q0 - dmin

    tau0_gam1 = q0_minus_dmax / v0mag
    tau1_gam1 = q0_minus_dmin / v0mag

    # Calculate alpha_c
    ac = erfcinv(gamma) # This works element-wise for numpy arrays

    # Calculate time bounds
    temp = ac * sv2

    tau0 = (-temp + q0_minus_dmax) / v0mag
    tau1 = ( temp + q0_minus_dmin) / v0mag

    if verbose:
        if gamma.ndim == 0:
            print(f"  gamma = {gamma}  tau0 = {tau0}  tau1 = {tau1}")
        else:
            for i in range(gamma.size):
                g = gamma.flat[i]
                t0 = tau0.flat[i]
                t1 = tau1.flat[i]
                print(f"  gamma = {g}  tau0 = {t0}  tau1 = {t1}")

    return tau0, tau1, tau0_gam1, tau1_gam1
