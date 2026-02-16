import numpy as np

def jacobian_equinoctial_to_cartesian(E, X, fr=1, mu=3.986004418e5):
    """
    Calculate the Jacobian, J = dX/dE, between the equinoctial elements and cartesian state.

    Parameters
    ----------
    E : array_like
        Equinoctial state vector (km) [6x1]
        (n, af, ag, chi, psi, lM)
    X : array_like
        Cartesian state vector (km, km/s) [6x1]
        (r, v)
    fr : int, optional
        Equinoctial element retrograde factor, by default 1
    mu : float, optional
        Gravitational constant, by default 3.986004418e5

    Returns
    -------
    ndarray
        Jacobian matrix dX/dE [6x6]
    """

    # Place Vectors in correct format
    E = np.array(E).flatten()
    X = np.array(X).flatten()

    if E.size != 6 or X.size != 6:
        raise ValueError("E and X must be 6-element vectors.")

    # Extract the elements
    n = E[0]
    af = E[1]
    ag = E[2]
    chi = E[3]
    psi = E[4]
    # lM = E[5]

    rvec = X[0:3]
    vvec = X[3:6]

    r2 = np.dot(rvec, rvec)
    r = np.sqrt(r2)
    r3 = r2 * r

    # Aux quantities
    n2 = n**2
    a3 = mu / n2
    a = np.cbrt(a3) # Using cbrt
    A = n * a**2

    ag2 = ag**2
    af2 = af**2
    B = np.sqrt(1 - ag2 - af2)
    Bp1 = B + 1

    # chi2 = chi**2
    # psi2 = psi**2
    chi2 = chi * chi
    psi2 = psi * psi
    C = 1 + chi2 + psi2

    # fhat and ghat
    # Note: Matlab code uses element-wise division by C for the whole vector

    fhat = np.array([
        1 - chi2 + psi2,
        2 * chi * psi,
        -2 * fr * chi
    ]) / C

    ghat = np.array([
        2 * fr * chi * psi,
        (1 + chi2 - psi2) * fr,
        2 * psi
    ]) / C

    what = np.array([
        2 * chi,
        -2 * psi,
        (1 - chi2 - psi2) * fr
    ]) / C

    # X, Y, Xdot, and Ydot
    X_val = np.dot(fhat, rvec)
    Y_val = np.dot(ghat, rvec)

    Xd = np.dot(fhat, vvec)
    Yd = np.dot(ghat, vvec)

    # Partials of X, Y, Xdot, Ydot
    AB = A * B
    nBp1 = n * Bp1
    Aor3 = A / r3

    dXdaf =  ag * Xd / nBp1 + a * (Y_val * Xd / AB - 1)
    dYdaf =  ag * Yd / nBp1 - a * (X_val * Xd / AB)
    dXdag = -af * Xd / nBp1 + a * (Y_val * Yd / AB)
    dYdag = -af * Yd / nBp1 - a * (X_val * Yd / AB + 1)

    dXddaf =  a * Xd * Yd / AB - Aor3 * (a * ag * X_val / Bp1 + X_val * Y_val / B)
    dYddaf = -a * Xd * Xd / AB - Aor3 * (a * ag * Y_val / Bp1 - X_val * X_val / B)
    dXddag =  a * Yd * Yd / AB + Aor3 * (a * af * X_val / Bp1 - Y_val * Y_val / B)
    dYddag = -a * Xd * Yd / AB + Aor3 * (a * af * Y_val / Bp1 + X_val * Y_val / B)

    # Define the Jacobian, J = dX/dE
    J = np.full((6, 6), np.nan)

    # drvec/dn & dvvec/dn
    cv = 1.0 / (3 * n)
    cr = -2 * cv
    J[0:3, 0] = cr * rvec
    J[3:6, 0] = cv * vvec

    # drvec/daf & dvvec/daf
    J[0:3, 1] = dXdaf * fhat + dYdaf * ghat
    J[3:6, 1] = dXddaf * fhat + dYddaf * ghat

    # drvec/dag & dvvec/dag
    J[0:3, 2] = dXdag * fhat + dYdag * ghat
    J[3:6, 2] = dXddag * fhat + dYddag * ghat

    # drvec/dchi & dvvec/dchi
    cc = 2.0 / C
    J[0:3, 3] = cc * ( fr * psi * (Y_val * fhat - X_val * ghat) - X_val * what )
    J[3:6, 3] = cc * ( fr * psi * (Yd * fhat - Xd * ghat) - Xd * what )

    # drvec/dpsi & dvvec/dpsi
    J[0:3, 4] = cc * ( fr * chi * (X_val * ghat - Y_val * fhat) + Y_val * what )
    J[3:6, 4] = cc * ( fr * chi * (Xd * ghat - Yd * fhat) + Yd * what )

    # drvec/dlambdaM & dvvec/dlambdaM
    J[0:3, 5] = vvec / n
    J[3:6, 5] = (-n * a3 / r3) * rvec

    return J
