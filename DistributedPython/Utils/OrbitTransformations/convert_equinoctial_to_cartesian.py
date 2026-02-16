import numpy as np
from scipy.optimize import minimize_scalar
import warnings

def convert_equinoctial_to_cartesian(n, af, ag, chi, psi, lam0, T, fr=1, mu=3.986004418e5, Ftol=None, maxiter=100, errflag=2):
    """
    Convert equinoctial orbital elements to cartesian states for a set of offset times from epoch.

    Parameters
    ----------
    n : float
        Mean motion (rad/s)
    af : float
        Equinoctial element af
    ag : float
        Equinoctial element ag
    chi : float
        Equinoctial element chi
    psi : float
        Equinoctial element psi
    lam0 : float
        Mean longitude at epoch (rad)
    T : array_like
        Time offsets from epoch (s). Can be scalar or array.
    fr : int, optional
        Equinoctial element retrograde factor, by default 1
    mu : float, optional
        Gravitational constant, by default 3.986004418e5
    Ftol : float, optional
        Tolerance for F angle convergence, by default 100*eps(2*pi)
    maxiter : int, optional
        Max iterations for F convergence, by default 100
    errflag : int, optional
        Error flag (0, 1, 2), by default 2.
        0 => No error/warning
        1 => Warning
        2 => Error

    Returns
    -------
    tuple
        rvec : ndarray (3, N)
            Cartesian position vector
        vvec : ndarray (3, N)
            Cartesian velocity vector
        X : ndarray (N,)
        Y : ndarray (N,)
        Xdot : ndarray (N,)
        Ydot : ndarray (N,)
        F : ndarray (N,)
            Eccentric longitude
        cF : ndarray (N,)
            Cos(F)
        sF : ndarray (N,)
            Sin(F)
    """

    if Ftol is None:
        Ftol = 100 * np.finfo(float).eps * (2 * np.pi)

    # Ensure inputs are numpy arrays/scalars
    T = np.atleast_1d(T)

    # Semimajor axis and related quantities
    n2 = n**2
    a3 = mu / n2
    a = np.cbrt(a3) # Using cbrt for stability like nthroot in Matlab
    na = n * a

    # Mean longitudes
    lam = lam0 + n * T

    # (ag,af) quantities
    ag2 = ag**2
    af2 = af**2

    B = np.sqrt(1 - ag2 - af2)
    b = 1.0 / (1 + B)

    omag2b = 1 - ag2 * b
    omaf2b = 1 - af2 * b
    afagb = af * ag * b

    # Unit vectors (f,g)
    chi2 = chi**2
    psi2 = psi**2
    C = 1 + chi2 + psi2

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

    # Calculate states and output quantities for each ephemeris point
    NT = T.size

    X = np.full(NT, np.nan)
    Y = np.full(NT, np.nan)
    Xdot = np.full(NT, np.nan)
    Ydot = np.full(NT, np.nan)
    F = np.full(NT, np.nan)
    cF = np.full(NT, np.nan)
    sF = np.full(NT, np.nan)

    rvec = np.full((3, NT), np.nan)
    vvec = np.full((3, NT), np.nan)

    for nt in range(NT):
        F_val, converged, cF_val, sF_val = equinoctial_kepeq(
            lam[nt], af, ag, Ftol, maxiter, errflag
        )

        F[nt] = F_val
        cF[nt] = cF_val
        sF[nt] = sF_val

        # If converged (or ignoring error), compute the rest
        # Matlab version returns NaNs if F is empty/invalid?
        # Actually equinoctial_kepeq returns empty F if it fails and errflag==1
        # But here we handle scalar return.

        if not converged and errflag == 1:
            # If warning issued, result might be NaN or partial
            pass
        elif not converged and errflag == 2:
             # Should have raised error inside equinoctial_kepeq
             pass

        if np.isnan(F_val):
            continue

        X[nt] = a * (omag2b * cF[nt] + afagb * sF[nt] - af)
        Y[nt] = a * (omaf2b * sF[nt] + afagb * cF[nt] - ag)

        rvec[:, nt] = X[nt] * fhat + Y[nt] * ghat

        denom = (1 - af * cF[nt] - ag * sF[nt])
        if denom == 0:
             na2or = np.nan # Avoid division by zero
        else:
             na2or = na / denom

        Xdot[nt] = na2or * (afagb * cF[nt] - omag2b * sF[nt])
        Ydot[nt] = na2or * (omaf2b * cF[nt] - afagb * sF[nt])

        vvec[:, nt] = Xdot[nt] * fhat + Ydot[nt] * ghat

    # If input T was scalar, maybe return squeezed arrays?
    # But Matlab returns [3xNT].
    # Python convention might prefer (NT, 3) but let's stick to matching Matlab output shape (3, NT) for now
    # as it might be expected by porting code.

    return rvec, vvec, X, Y, Xdot, Ydot, F, cF, sF


def equinoctial_kepeq(lam, af, ag, Ftol, maxiter, errflag):
    """
    Solve Kepler's equation in equinoctial elements.
    """

    # Ensure mean longitude in range 0 <= lam < 2*pi
    lam = lam % (2 * np.pi)

    converged = False

    ecc2 = af**2 + ag**2

    if ecc2 >= 1:
        errtext = 'equinoctial_kepeq cannot process eccentricities >= 1'
        if errflag == 1:
            warnings.warn(errtext)
            return np.nan, False, np.nan, np.nan
        elif errflag == 2:
            raise ValueError(errtext)
        elif errflag != 0:
            raise ValueError('Invalid errflag value')
        return np.nan, False, np.nan, np.nan

    # Initial guess
    F = lam
    cF = np.cos(F)
    sF = np.sin(F)

    iter_count = 0
    still_looping = True

    while still_looping:
        top = F + ag * cF - af * sF - lam
        bot = 1 - ag * sF - af * cF

        if bot == 0:
             # Singularity
             break

        Fdel = top / bot

        F = F - Fdel
        cF = np.cos(F)
        sF = np.sin(F)

        absFdel = abs(Fdel)

        if absFdel < Ftol:
            converged = True
            still_looping = False
        elif iter_count >= maxiter or absFdel >= np.pi:
            still_looping = False
        else:
            iter_count += 1

    if not converged:
        # Fallback to optimization
        def KEP(FF):
            return abs(FF + ag * np.cos(FF) - af * np.sin(FF) - lam)

        res = minimize_scalar(KEP, bounds=(0, 2*np.pi), method='bounded', options={'xatol': Ftol, 'maxiter': max(maxiter, 500)})

        if res.success:
            converged = True
            F = res.x
            cF = np.cos(F)
            sF = np.sin(F)

        if not converged:
            errtext = 'equinoctial_kepeq failed to converge'
            if errflag == 1:
                warnings.warn(errtext)
                return np.nan, False, np.nan, np.nan
            elif errflag == 2:
                raise RuntimeError(errtext)
            elif errflag != 0:
                raise ValueError('Invalid errflag value')

    return F, converged, cF, sF
