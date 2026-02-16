import numpy as np
from DistributedPython.Utils.OrbitTransformations.convert_equinoctial_to_cartesian import convert_equinoctial_to_cartesian
from DistributedPython.Utils.OrbitTransformations.jacobian_equinoctial_to_cartesian import jacobian_equinoctial_to_cartesian

def jacobian_E0_to_Xt(T, E0, fr=1, mu=3.986004418e5, errflag=2):
    """
    Calculate the Jacobian of the cartesian state at time t with respect to the initial equinoctial elements.

    Parameters
    ----------
    T : array_like
        Time offsets from initial (s). [NT]
    E0 : array_like
        Equinoctial elements at initial time t0. [6x1]
        (n, af, ag, chi, psi, lM)
    fr : int, optional
        Equinoctial element retrograde factor, by default 1
    mu : float, optional
        Gravitational constant, by default 3.986004418e5
    errflag : int, optional
        Error flag (0, 1, 2), by default 2.

    Returns
    -------
    tuple
        JT : ndarray (NT, 6, 6)
            EpochEquinoctial-to-EphemerisCartesian transformations
            Note: Python convention (NT, 6, 6) vs Matlab (6, 6, NT)
        XT : ndarray (6, NT)
            Center-of-expansion cartesian state ephemeris (km & km/s)
    """

    # Defaults handled by function signature or inside logic

    T = np.atleast_1d(T)
    NT = T.size

    E0 = np.array(E0).flatten()
    if E0.size != 6:
        raise ValueError("E0 must be a 6-element vector.")

    # Calculate X(t)
    rT, vT, _, _, _, _, _, _, _ = convert_equinoctial_to_cartesian(
        E0[0], E0[1], E0[2], E0[3], E0[4], E0[5], T,
        fr=fr, mu=mu, errflag=errflag
    )

    # XT = [rT; vT] in Matlab -> shape (6, NT)
    XT = np.vstack((rT, vT))

    # Initialize dE(t)/dE(t0) STM
    phi = np.eye(6)

    # Initialize E(t)
    ET = E0.copy()

    # Initialize output Jacobian array
    # Python convention: (NT, 6, 6) typically, but let's see.
    # Matlab: (6, 6, NT).
    # If I use (NT, 6, 6), I should document it.
    # The prompt asks to convert code. Usually Python uses batch dimension first.
    # Memory says: "When handling stacked matrices... convention is batch-first (N, 3, 3) or (N, 6, 6)".
    # So I will use (NT, 6, 6).

    JT = np.full((NT, 6, 6), np.nan)

    # Loop over times and calculate output
    for nT in range(NT):
        # Calculate the equinoctial mean longitude at this time
        ET[5] = E0[5] + T[nT] * E0[0]

        # Define off-diagonal dE(t)/dE(t0) STM element
        phi[5, 0] = T[nT]

        # Calculate dX(t)/dE(t) STM
        # XT[:, nT] is the state at time nT
        J = jacobian_equinoctial_to_cartesian(ET, XT[:, nT], fr=fr, mu=mu)

        # Calculate the dX(t)/dE0 STM
        # JT(:,:,nT) = J*phi
        JT[nT, :, :] = np.dot(J, phi)

    return JT, XT
