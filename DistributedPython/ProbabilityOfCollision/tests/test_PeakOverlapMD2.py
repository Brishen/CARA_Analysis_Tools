import unittest
import numpy as np
from DistributedPython.ProbabilityOfCollision.Utils.PeakOverlapMD2 import PeakOverlapMD2
from DistributedPython.Utils.OrbitTransformations.convert_cartesian_to_equinoctial import convert_cartesian_to_equinoctial
from DistributedPython.ProbabilityOfCollision.Utils.jacobian_E0_to_Xt import jacobian_E0_to_Xt

class TestPeakOverlapMD2(unittest.TestCase):

    def test_peak_overlap_md2_basic(self):
        # Setup a simple conjunction case
        mu = 3.986004418e5

        # Object 1: Circular LEO
        r1 = np.array([7000.0, 0.0, 0.0])
        v1 = np.array([0.0, np.sqrt(mu/7000.0), 0.0])

        # Object 2: Circular LEO, slightly offset
        r2 = np.array([7000.1, 0.1, 0.0])
        v2 = np.array([0.0, np.sqrt(mu/7000.1), 0.0])

        # Initial time
        t10 = 0.0
        t20 = 0.0

        # Convert to Equinoctial
        res1 = convert_cartesian_to_equinoctial(r1, v1, mu=mu)
        Eb10 = np.array([res1[1], res1[2], res1[3], res1[4], res1[5], res1[6]])

        res2 = convert_cartesian_to_equinoctial(r2, v2, mu=mu)
        Eb20 = np.array([res2[1], res2[2], res2[3], res2[4], res2[5], res2[6]])

        # Covariances
        Qb10 = np.diag([1e-14, 1e-8, 1e-8, 1e-8, 1e-8, 1e-6])
        Qb20 = np.diag([1e-14, 1e-8, 1e-8, 1e-8, 1e-8, 1e-6])

        HBR = 10.0 # meters
        EMD2 = 1 # Mode 1

        params = {'verbose': False}

        t = 0.0

        MD2, Xu, Ps, Asdet, Asinv, POPconverged, aux = PeakOverlapMD2(
            t, t10, Eb10, Qb10, t20, Eb20, Qb20, HBR, EMD2, params
        )

        self.assertTrue(POPconverged)
        self.assertFalse(np.isnan(MD2))
        self.assertIsNotNone(Xu)
        self.assertIsNotNone(Ps)
        self.assertIsNotNone(Asdet)
        self.assertIsNotNone(Asinv)
        self.assertTrue(len(aux) > 0)

    def test_peak_overlap_md2_xc(self):
         # Test with XC processing enabled
        mu = 3.986004418e5
        r1 = np.array([7000.0, 0.0, 0.0])
        v1 = np.array([0.0, np.sqrt(mu/7000.0), 0.0])
        r2 = np.array([7000.1, 0.1, 0.0])
        v2 = np.array([0.0, np.sqrt(mu/7000.1), 0.0])
        t10 = 0.0; t20 = 0.0

        res1 = convert_cartesian_to_equinoctial(r1, v1, mu=mu)
        Eb10 = np.array([res1[1], res1[2], res1[3], res1[4], res1[5], res1[6]])
        res2 = convert_cartesian_to_equinoctial(r2, v2, mu=mu)
        Eb20 = np.array([res2[1], res2[2], res2[3], res2[4], res2[5], res2[6]])
        Qb10 = np.diag([1e-14, 1e-8, 1e-8, 1e-8, 1e-8, 1e-6])
        Qb20 = np.diag([1e-14, 1e-8, 1e-8, 1e-8, 1e-8, 1e-6])

        HBR = 10.0
        EMD2 = 1

        # Mock XC params
        params = {
            'verbose': False,
            'XCprocessing': True,
            'GEp': np.ones(6),
            'GEs': np.ones(6),
            'sigpXsigs': 1e-6
        }

        t = 0.0

        MD2, Xu, Ps, Asdet, Asinv, POPconverged, aux = PeakOverlapMD2(
            t, t10, Eb10, Qb10, t20, Eb20, Qb20, HBR, EMD2, params
        )

        self.assertTrue(POPconverged)
        self.assertFalse(np.isnan(MD2))

if __name__ == '__main__':
    unittest.main()
