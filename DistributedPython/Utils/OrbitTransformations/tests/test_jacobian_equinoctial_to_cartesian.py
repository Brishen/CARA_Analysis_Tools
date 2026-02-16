import numpy as np
import pytest
from DistributedPython.Utils.OrbitTransformations.convert_equinoctial_to_cartesian import convert_equinoctial_to_cartesian
from DistributedPython.Utils.OrbitTransformations.jacobian_equinoctial_to_cartesian import jacobian_equinoctial_to_cartesian

class TestJacobianEquinoctialToCartesian:

    def test_jacobian_finite_difference(self):
        # Initial parameters
        n = 0.000618165872034742
        af = -0.186939977644424
        ag = 0.0251358447064123
        chi = -0.206130724502682
        psi = -2.36641891694603
        lam0 = -1.08358461740347
        T = 0

        E = np.array([n, af, ag, chi, psi, lam0])
        fr = 1
        mu = 3.986004418e5

        # Calculate X using the conversion function
        rvec, vvec, _, _, _, _, _, _, _ = convert_equinoctial_to_cartesian(
            n, af, ag, chi, psi, lam0, T, fr=fr, mu=mu
        )
        X = np.concatenate((rvec.flatten(), vvec.flatten()))

        # Calculate Jacobian
        J = jacobian_equinoctial_to_cartesian(E, X, fr=fr, mu=mu)

        # Finite difference
        epsilon = 1e-7
        J_fd = np.zeros((6, 6))

        for i in range(6):
            E_plus = E.copy()
            E_plus[i] += epsilon

            E_minus = E.copy()
            E_minus[i] -= epsilon

            # n, af, ag, chi, psi, lam0
            r_plus, v_plus, _, _, _, _, _, _, _ = convert_equinoctial_to_cartesian(
                E_plus[0], E_plus[1], E_plus[2], E_plus[3], E_plus[4], E_plus[5], T, fr=fr, mu=mu
            )
            X_plus = np.concatenate((r_plus.flatten(), v_plus.flatten()))

            r_minus, v_minus, _, _, _, _, _, _, _ = convert_equinoctial_to_cartesian(
                E_minus[0], E_minus[1], E_minus[2], E_minus[3], E_minus[4], E_minus[5], T, fr=fr, mu=mu
            )
            X_minus = np.concatenate((r_minus.flatten(), v_minus.flatten()))

            J_fd[:, i] = (X_plus - X_minus) / (2 * epsilon)

        np.testing.assert_allclose(J, J_fd, rtol=1e-4, atol=1e-4)
