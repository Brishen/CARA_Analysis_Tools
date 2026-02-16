import numpy as np
import warnings

def convert_cartesian_to_equinoctial(rvec, vvec, fr=1, mu=3.986004418e5, issue_warnings=True):
    """
    Convert cartesian state (r,v) to the equinoctial orbital elements (a,n,af,ag,chi,psi,lambdaM,F).

    Args:
        rvec (array-like): Cartesian position vector (km) [3x1]
        vvec (array-like): Cartesian velocity vector (km) [3x1]
        fr (int, optional): Equinoctial element retrograde factor. Defaults to 1.
        mu (float, optional): Gravitational constant. Defaults to 3.986004418e5.
        issue_warnings (bool, optional): Display warning messages. Defaults to True.

    Returns:
        tuple: (a, n, af, ag, chi, psi, lM, F) or (None, None, None, None, None, None, None, None) on failure.
    """

    if abs(fr) != 1:
        if issue_warnings:
            warnings.warn('convert_cartesian_to_equinoctial:InvalidRetrogradeFactor: fr must be either +1, -1, or not passed in (default = +1)')
        return None, None, None, None, None, None, None, None

    rvec = np.asarray(rvec).flatten()
    vvec = np.asarray(vvec).flatten()

    if rvec.size != 3 or vvec.size != 3:
        raise ValueError("rvec and vvec must be 3-element vectors")

    # Calculate equinoctial elements using equations for bound orbits.
    r = np.linalg.norm(rvec)
    v2 = np.dot(vvec, vvec)

    # Semi-major axis
    denom = 2 * mu - v2 * r
    if denom == 0:
         a = np.inf
    else:
         a = mu * r / denom

    # Issue warning and return for unbound orbits: a <= 0 implies E >= 0
    # Also catch very small positive a which might be numerical noise for 0
    if a <= 1e-5 or np.isinf(a):
        if issue_warnings:
            warnings.warn('convert_cartesian_to_equinoctial:UnboundOrbit: Cannot process unbound orbit with infinite or nonpositive semimajor axis (a)')
        return None, None, None, None, None, None, None, None

    rdv = np.dot(rvec, vvec)

    # rxv cross product
    rcv = np.cross(rvec, vvec)

    # Mean motion
    n = np.sqrt(mu / a**3)

    # Eccentricity vector
    evec = (1 / mu) * ((v2 - mu / r) * rvec - rdv * vvec)

    if np.dot(evec, evec) >= 1:
        if issue_warnings:
            warnings.warn('convert_cartesian_to_equinoctial:UnboundOrbit: Cannot process unbound orbit with ecc^2 = evec.evec >= 1')
        return None, None, None, None, None, None, None, None

    # Calculate chi and psi elements
    what = rcv / np.linalg.norm(rcv)

    # Check for singularity (1 + fr * what[2] must be > 0)
    # If fr=1, singularity at i=180 (what[2]=-1) -> 1 + 1*(-1) = 0
    # If fr=-1, singularity at i=0 (what[2]=1) -> 1 + (-1)*(1) = 0
    if 1 + fr * what[2] <= 1e-10:
        if issue_warnings:
            warnings.warn('convert_cartesian_to_equinoctial:EquatorialRetrogradeOrbit: Cannot process equatorial retrograde orbits (i = 180) without flag')
        return None, None, None, None, None, None, None, None

    cpden = (1 + fr * what[2])
    chi = what[0] / cpden
    psi = -what[1] / cpden

    chi2 = chi**2
    psi2 = psi**2
    C = 1 + chi2 + psi2

    # Calculate f and g unit vectors
    fhat = np.array([1 - chi2 + psi2, 2 * chi * psi, -2 * fr * chi]) / C
    ghat = np.array([2 * fr * chi * psi, (1 + chi2 - psi2) * fr, 2 * psi]) / C

    af = np.dot(fhat, evec)
    ag = np.dot(ghat, evec)

    af2 = af**2
    ag2 = ag**2
    ec2 = ag2 + af2

    if ec2 >= 1:
        if issue_warnings:
            warnings.warn('convert_cartesian_to_equinoctial:UnboundOrbit: Cannot process unbound orbit with ecc^2 = af^2 + ag^2 >= 1')
        return None, None, None, None, None, None, None, None

    X = np.dot(fhat, rvec)
    Y = np.dot(ghat, rvec)

    safg = np.sqrt(1 - ag2 - af2)
    b = 1 / (1 + safg)
    Fden = a * safg

    bagaf = b * ag * af

    sinF = ag + ((1 - ag2 * b) * Y - bagaf * X) / Fden
    cosF = af + ((1 - af2 * b) * X - bagaf * Y) / Fden
    F = np.arctan2(sinF, cosF)

    lM = F + ag * cosF - af * sinF

    # MATLAB returns lM (lambdaM) as 7th output, F as 8th.
    return a, n, af, ag, chi, psi, lM, F
