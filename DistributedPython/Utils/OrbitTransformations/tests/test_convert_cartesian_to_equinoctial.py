import unittest
import numpy as np
import warnings
from DistributedPython.Utils.OrbitTransformations.convert_cartesian_to_equinoctial import convert_cartesian_to_equinoctial

class TestConvertCartesianToEquinoctial(unittest.TestCase):
    def test_circular_orbit(self):
        # Circular orbit in XY plane
        rvec = np.array([7000.0, 0.0, 0.0])
        # Circular velocity v = sqrt(mu/r)
        mu = 3.986004418e5
        v_circ = np.sqrt(mu / 7000.0)
        vvec = np.array([0.0, v_circ, 0.0])

        a, n, af, ag, chi, psi, lM, F = convert_cartesian_to_equinoctial(rvec, vvec, mu=mu)

        self.assertIsNotNone(a)
        self.assertAlmostEqual(a, 7000.0, delta=1.0)
        self.assertAlmostEqual(af, 0.0, delta=1e-3)
        self.assertAlmostEqual(ag, 0.0, delta=1e-3)
        self.assertAlmostEqual(chi, 0.0, delta=1e-3)
        self.assertAlmostEqual(psi, 0.0, delta=1e-3)

    def test_unbound_orbit(self):
        rvec = np.array([7000.0, 0.0, 0.0])
        vvec = np.array([0.0, 15.0, 0.0]) # hyperbolic

        # Suppress warnings for test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a, n, af, ag, chi, psi, lM, F = convert_cartesian_to_equinoctial(rvec, vvec, issue_warnings=False)
        self.assertIsNone(a)

    def test_retrograde_orbit(self):
        # Retrograde circular orbit in XY plane
        # Velocity in -Y direction
        rvec = np.array([7000.0, 0.0, 0.0])
        mu = 3.986004418e5
        v_circ = np.sqrt(mu / 7000.0)
        vvec = np.array([0.0, -v_circ, 0.0])

        # Should fail if fr default (1) is used for retrograde orbit (i=180)
        # because i=180 is a singularity for fr=1 (prograde equinoctial elements)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a, n, af, ag, chi, psi, lM, F = convert_cartesian_to_equinoctial(rvec, vvec, fr=1, mu=mu)
        self.assertIsNone(a)

        # With fr=-1, it should work (using retrograde equinoctial elements)
        # This was previously failing due to a bug in the singularity check, now fixed.
        a, n, af, ag, chi, psi, lM, F = convert_cartesian_to_equinoctial(rvec, vvec, fr=-1, mu=mu)
        self.assertIsNotNone(a)
        self.assertAlmostEqual(a, 7000.0, delta=1.0)
        # For retrograde equatorial, i=180.
        # what = [0, 0, -1].
        # cpden = 1 + (-1)*(-1) = 2.
        # chi = 0/2 = 0. psi = 0/2 = 0.
        # fhat = [1, 0, 0]. ghat = [0, 1, 0].
        # af, ag should be 0.
        self.assertAlmostEqual(af, 0.0, delta=1e-3)
        self.assertAlmostEqual(ag, 0.0, delta=1e-3)
        self.assertAlmostEqual(chi, 0.0, delta=1e-3)
        self.assertAlmostEqual(psi, 0.0, delta=1e-3)
