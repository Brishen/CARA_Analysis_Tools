import unittest
import numpy as np
from DistributedPython.ProbabilityOfCollision.Pc2D_Hall import Pc2D_Hall

class TestPc2D_Hall(unittest.TestCase):

    def test_pc2d_hall_basic(self):
        # Setup a simple conjunction case
        mu = 3.986004418e5

        # Object 1
        r1 = np.array([7000.0, 0.0, 0.0]) * 1000 # meters
        v1 = np.array([0.0, np.sqrt(mu/7000.0), 0.0]) * 1000 # m/s

        # Object 2: intersecting but slightly offset (100m) to avoid 0/0 issues
        r2 = np.array([7000.0, 0.1, 0.0]) * 1000 # meters
        v2 = np.array([0.0, 0.0, np.sqrt(mu/7000.0)]) * 1000 # m/s

        # Small covariance
        C1 = np.eye(6) * 100.0
        C2 = np.eye(6) * 100.0

        HBR = 10.0 # meters

        params = {'verbose': False}

        Pc, out = Pc2D_Hall(r1, v1, C1, r2, v2, C2, HBR, params)

        self.assertIsNotNone(Pc)
        self.assertFalse(np.isnan(Pc), f"Pc is NaN. out: {out}")
        self.assertTrue(0.0 <= Pc <= 1.0)

    def test_pc2d_hall_miss(self):
        # Setup a miss case
        mu = 3.986004418e5

        r1 = np.array([7000.0, 0.0, 0.0]) * 1000
        v1 = np.array([0.0, np.sqrt(mu/7000.0), 0.0]) * 1000

        # Object 2: far away
        r2 = np.array([7100.0, 0.0, 0.0]) * 1000
        v2 = np.array([0.0, 0.0, np.sqrt(mu/7100.0)]) * 1000

        C1 = np.eye(6) * 100.0
        C2 = np.eye(6) * 100.0

        HBR = 10.0

        params = {'verbose': False}

        Pc, out = Pc2D_Hall(r1, v1, C1, r2, v2, C2, HBR, params)

        self.assertIsNotNone(Pc)
        self.assertFalse(np.isnan(Pc))
        self.assertTrue(Pc < 1e-10)

if __name__ == '__main__':
    unittest.main()
