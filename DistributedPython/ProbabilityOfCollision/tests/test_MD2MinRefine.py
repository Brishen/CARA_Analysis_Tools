import unittest
import numpy as np
from DistributedPython.ProbabilityOfCollision.Utils.MD2MinRefine import MD2MinRefine
from DistributedPython.Utils.OrbitTransformations.convert_cartesian_to_equinoctial import convert_cartesian_to_equinoctial

class TestMD2MinRefine(unittest.TestCase):

    def test_md2_min_refine_basic(self):
        # Setup a simple conjunction case
        mu = 3.986004418e5

        # Object 1
        r1 = np.array([7000.0, 0.0, 0.0])
        v1 = np.array([0.0, np.sqrt(mu/7000.0), 0.0])

        # Object 2: intersecting at t=0
        r2 = np.array([7000.0, 0.0, 0.0])
        v2 = np.array([0.0, 0.0, np.sqrt(mu/7000.0)])

        # Convert to Equinoctial
        res1 = convert_cartesian_to_equinoctial(r1, v1, mu=mu)
        Eb10 = np.array([res1[1], res1[2], res1[3], res1[4], res1[5], res1[6]])

        res2 = convert_cartesian_to_equinoctial(r2, v2, mu=mu)
        Eb20 = np.array([res2[1], res2[2], res2[3], res2[4], res2[5], res2[6]])

        Qb10 = np.diag([1e-14, 1e-8, 1e-8, 1e-8, 1e-8, 1e-6])
        Qb20 = np.diag([1e-14, 1e-8, 1e-8, 1e-8, 1e-8, 1e-6])

        HBR = 10.0
        POPPAR = {'verbose': False}

        # Call MD2MinRefine
        tmd = 0.0 # Start search at TCA=0
        tsg = 1.0 # Initial sigma guess
        dtsg = 0.1
        ttol = 1e-3
        itermax = 10
        findmin = True

        out = MD2MinRefine(tmd, tsg, dtsg, ttol, itermax, findmin, Eb10, Qb10, Eb20, Qb20, HBR, POPPAR)

        # We expect convergence and tmd close to 0
        self.assertTrue(out['MD2converged'])
        self.assertFalse(np.isnan(out['ymd']))
        self.assertTrue(abs(out['tmd']) < 1.0)

if __name__ == '__main__':
    unittest.main()
