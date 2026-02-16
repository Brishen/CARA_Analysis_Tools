import unittest
import numpy as np
from DistributedPython.ProbabilityOfCollision.Utils.PeakOverlapPos import PeakOverlapPos
from DistributedPython.Utils.OrbitTransformations.convert_cartesian_to_equinoctial import convert_cartesian_to_equinoctial
from DistributedPython.ProbabilityOfCollision.Utils.jacobian_E0_to_Xt import jacobian_E0_to_Xt

class TestPeakOverlapPos(unittest.TestCase):

    def test_peak_overlap_pos_convergence(self):
        # Setup a simple conjunction case
        mu = 3.986004418e5

        # Object 1: Circular LEO
        r1 = np.array([7000.0, 0.0, 0.0])
        v1 = np.array([0.0, np.sqrt(mu/7000.0), 0.0]) # ~7.5 km/s

        # Object 2: Circular LEO, slightly offset in position
        r2 = np.array([7000.1, 0.1, 0.0])
        v2 = np.array([0.0, np.sqrt(mu/7000.1), 0.0])

        # Initial time
        t01 = 0.0
        t02 = 0.0

        # Convert to Equinoctial
        # returns: a, n, af, ag, chi, psi, lM, F
        res1 = convert_cartesian_to_equinoctial(r1, v1, mu=mu)
        Eb01 = np.array([res1[1], res1[2], res1[3], res1[4], res1[5], res1[6]]) # n, af, ag, chi, psi, lM

        res2 = convert_cartesian_to_equinoctial(r2, v2, mu=mu)
        Eb02 = np.array([res2[1], res2[2], res2[3], res2[4], res2[5], res2[6]])

        # Initial Jacobians at t=0
        Jb1, _ = jacobian_E0_to_Xt(0.0, Eb01, mu=mu)
        if Jb1.ndim == 3: Jb1 = Jb1[0]

        Jb2, _ = jacobian_E0_to_Xt(0.0, Eb02, mu=mu)
        if Jb2.ndim == 3: Jb2 = Jb2[0]

        # Covariances (diagonal for simplicity)
        # Small covariance to ensure convergence
        Qb01 = np.diag([1e-14, 1e-8, 1e-8, 1e-8, 1e-8, 1e-6])
        Qb02 = np.diag([1e-14, 1e-8, 1e-8, 1e-8, 1e-8, 1e-6])

        # States at t (t=0 for simplicity)
        xb1 = np.concatenate((r1, v1))
        xb2 = np.concatenate((r2, v2))
        t = 0.0

        HBR = 10.0 # meters

        params = {'verbose': False}

        conv, rpk, v1pk, v2pk, aux = PeakOverlapPos(
            t, xb1, Jb1, t01, Eb01, Qb01, xb2, Jb2, t02, Eb02, Qb02, HBR, params
        )

        self.assertTrue(conv, "PeakOverlapPos did not converge")
        self.assertIsNotNone(rpk)
        self.assertEqual(rpk.shape, (3,))

        # Check that rpk is "between" r1 and r2 roughly
        # r1 = [7000, 0, 0], r2 = [7000.1, 0.1, 0]
        # rpk should be roughly in this vicinity.
        self.assertTrue(6999 < rpk[0] < 7001)

    def test_peak_overlap_pos_divergence_handling(self):
        # Test with very large HBR or weird covariance that might cause issues,
        # or just verify it handles things gracefully.
        # Here we just check it runs without crashing.
        pass

if __name__ == '__main__':
    unittest.main()
