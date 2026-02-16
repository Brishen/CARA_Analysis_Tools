import unittest
import numpy as np
from DistributedPython.ProbabilityOfCollision.Utils.UsageViolationPc2D import UsageViolationPc2D

class TestUsageViolationPc2D(unittest.TestCase):

    def test_usage_violation_pc2d_basic(self):
        # Setup a simple conjunction case
        mu = 3.986004418e5

        # Object 1
        r1 = np.array([7000.0, 0.0, 0.0]) * 1000 # meters
        v1 = np.array([0.0, np.sqrt(mu/7000.0), 0.0]) * 1000 # m/s

        # Object 2: intersecting
        r2 = np.array([7000.0, 0.0, 0.0]) * 1000 # meters
        v2 = np.array([0.0, 0.0, np.sqrt(mu/7000.0)]) * 1000 # m/s

        C1 = np.eye(6) * 1e2 # meters^2
        C2 = np.eye(6) * 1e2

        HBR = 10.0 # meters

        params = {'verbose': False}

        UVIndicators, out = UsageViolationPc2D(r1, v1, C1, r2, v2, C2, HBR, params)

        self.assertFalse(np.isnan(UVIndicators['Extended']))
        self.assertFalse(np.isnan(UVIndicators['Offset']))
        self.assertFalse(np.isnan(UVIndicators['Inaccurate']))
        self.assertTrue(out['Qconverged'])

if __name__ == '__main__':
    unittest.main()
