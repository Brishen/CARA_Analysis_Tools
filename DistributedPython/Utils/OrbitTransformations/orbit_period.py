import numpy as np

def orbit_period(r0vec, v0vec, GM=3.986004418e14):
    """
    Calculate orbital period assuming two-body motion.

    Args:
        r0vec (array-like): Position vector (m or km)
        v0vec (array-like): Velocity vector (m/s or km/s)
        GM (float, optional): Gravitational constant. Defaults to 3.986004418e14 (m^3/s^2).

    Returns:
        float: Orbital period (s)
    """
    r0vec = np.asarray(r0vec).flatten()
    v0vec = np.asarray(v0vec).flatten()

    r0 = np.linalg.norm(r0vec)
    v02 = np.dot(v0vec, v0vec)

    # Vis-viva equation: v^2 = GM * (2/r - 1/a) => 1/a = 2/r - v^2/GM
    # Energy = -GM/2a = v^2/2 - GM/r
    # beta = 1/a * GM = 2*GM/r - v^2 ??
    # MATLAB code: beta = 2*GM/r0 - v02
    # This means beta = GM/a.
    # period = 2*pi * sqrt(a^3/GM) = 2*pi * (GM/beta)^(3/2) / sqrt(GM) ??
    # period = 2*pi * GM * beta^(-1.5) ??
    # Let's check: beta = GM/a. beta^(-1.5) = (a/GM)^(1.5) = a^1.5 / GM^1.5
    # GM * beta^(-1.5) = GM * a^1.5 / GM^1.5 = a^1.5 / GM^0.5 = sqrt(a^3/GM).
    # Correct.

    beta = 2*GM/r0 - v02
    if beta <= 0:
        return np.inf # Unbound orbit

    period = (2*np.pi) * GM * beta**(-1.5)
    return period
