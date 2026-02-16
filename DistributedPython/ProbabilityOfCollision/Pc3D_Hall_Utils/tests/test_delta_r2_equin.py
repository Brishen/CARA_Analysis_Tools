import numpy as np
import pytest
from DistributedPython.ProbabilityOfCollision.Pc3D_Hall_Utils.delta_r2_equin import delta_r2_equin

def test_delta_r2_equin():
    # Simple test case: two objects in circular orbits, same plane, different radii
    # Object 1: r = 7000 km
    # Object 2: r = 8000 km
    # Distance should be 1000 km, squared = 1e6 km^2 = 1e12 m^2

    # Equinoctial elements for circular orbit:
    # n = sqrt(mu/a^3)
    # af = 0, ag = 0 (circular)
    # chi = 0, psi = 0 (equatorial)
    # lM = 0

    GM = 398600.4418 * 1e9 # m^3/s^2
    r1 = 7000e3 # m
    r2 = 8000e3 # m

    n1 = np.sqrt(GM / r1**3)
    n2 = np.sqrt(GM / r2**3)

    Ts = np.array([0.0])

    dr2 = delta_r2_equin(Ts,
        n1, 0, 0, 0, 0, 0,
        n2, 0, 0, 0, 0, 0,
        GM)

    expected_dr = r2 - r1
    expected_dr2 = expected_dr**2

    np.testing.assert_allclose(dr2, expected_dr2, rtol=1e-7)

    # Test vectorized time
    Ts = np.array([0.0, 10.0])
    # At t=0, dist is 1000km. At t=10, they move slightly, dist changes.
    dr2_vec = delta_r2_equin(Ts,
        n1, 0, 0, 0, 0, 0,
        n2, 0, 0, 0, 0, 0,
        GM)

    assert dr2_vec.shape == (2,)
    np.testing.assert_allclose(dr2_vec[0], expected_dr2, rtol=1e-7)
