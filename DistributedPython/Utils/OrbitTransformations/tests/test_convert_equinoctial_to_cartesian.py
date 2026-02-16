import numpy as np
import pytest
from DistributedPython.Utils.OrbitTransformations.convert_equinoctial_to_cartesian import convert_equinoctial_to_cartesian

class TestConvertEquinoctialToCartesian:

    # Values from convert_equinoctial_to_cartesian_UnitTest.m
    # ver = {[0.000618165872034742, -0.186939977644424, 0.0251358447064123, -0.206130724502682, -2.36641891694603, -1.08358461740347, 0]};
    # verRes = {[6513.711946044443, 6882.748803674651, 6438.350633995484, 4.902960531245754, -1.518827797965108, -1.983803852683226]};

    def test_known_case(self):
        # n, af, ag, chi, psi, lam0, T
        # T is 0 in ver[6]

        n = 0.000618165872034742
        af = -0.186939977644424
        ag = 0.0251358447064123
        chi = -0.206130724502682
        psi = -2.36641891694603
        lam0 = -1.08358461740347
        T = 0

        rvec, vvec, X, Y, Xdot, Ydot, F, cF, sF = convert_equinoctial_to_cartesian(
            n, af, ag, chi, psi, lam0, T
        )

        expected_r = np.array([6513.711946044443, 6882.748803674651, 6438.350633995484])
        expected_v = np.array([4.902960531245754, -1.518827797965108, -1.983803852683226])

        np.testing.assert_allclose(rvec.flatten(), expected_r, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(vvec.flatten(), expected_v, rtol=1e-5, atol=1e-5)

    def test_vectorized_input(self):
        n = 0.000618165872034742
        af = -0.186939977644424
        ag = 0.0251358447064123
        chi = -0.206130724502682
        psi = -2.36641891694603
        lam0 = -1.08358461740347
        T = np.array([0, 10, 20])

        rvec, vvec, X, Y, Xdot, Ydot, F, cF, sF = convert_equinoctial_to_cartesian(
            n, af, ag, chi, psi, lam0, T
        )

        assert rvec.shape == (3, 3)
        assert vvec.shape == (3, 3)
        assert F.shape == (3,)
