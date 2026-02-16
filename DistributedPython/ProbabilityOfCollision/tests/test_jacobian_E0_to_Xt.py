import numpy as np
import pytest
from DistributedPython.ProbabilityOfCollision.Utils.jacobian_E0_to_Xt import jacobian_E0_to_Xt

class TestJacobianE0ToXt:

    def test_jacobian_E0_to_Xt(self):
        # Initial parameters
        n = 0.000618165872034742
        af = -0.186939977644424
        ag = 0.0251358447064123
        chi = -0.206130724502682
        psi = -2.36641891694603
        lam0 = -1.08358461740347

        E0 = np.array([n, af, ag, chi, psi, lam0])
        T = np.array([0, 10, 20])
        fr = 1
        mu = 3.986004418e5

        JT, XT = jacobian_E0_to_Xt(T, E0, fr=fr, mu=mu)

        assert JT.shape == (3, 6, 6)
        assert XT.shape == (6, 3)

        # At T=0, phi should be identity (except phi[5,0]=0).
        # So JT should be close to J at T=0.

        # Verify derivative wrt mean longitude (lam0)
        # Check finite difference for first time step

        epsilon = 1e-7
        J_fd = np.zeros((6, 6))

        for i in range(6):
            E_plus = E0.copy()
            E_plus[i] += epsilon

            E_minus = E0.copy()
            E_minus[i] -= epsilon

            # Use T=0
            _, X_plus = jacobian_E0_to_Xt(np.array([0]), E_plus, fr=fr, mu=mu)
            _, X_minus = jacobian_E0_to_Xt(np.array([0]), E_minus, fr=fr, mu=mu)

            J_fd[:, i] = (X_plus[:, 0] - X_minus[:, 0]) / (2 * epsilon)

        np.testing.assert_allclose(JT[0], J_fd, rtol=1e-4, atol=1e-4)
