import unittest
import numpy as np
from DistributedPython.ProbabilityOfCollision.PcCircle import PcCircle

class TestPcCircle(unittest.TestCase):
    def test_basic_validation(self):
        # Create some dummy data
        r1 = np.array([[1000.0, 0.0, 0.0]])
        v1 = np.array([[0.0, 7.0, 0.0]])
        cov1 = np.eye(3).reshape(1, 9)

        r2 = np.array([[1010.0, 0.0, 0.0]])
        v2 = np.array([[0.0, 7.1, 0.0]])
        cov2 = np.eye(3).reshape(1, 9)

        HBR = 5.0

        Pc, out = PcCircle(r1, v1, cov1, r2, v2, cov2, HBR)

        self.assertEqual(len(Pc), 1)
        self.assertTrue(out['IsPosDef'][0])
        self.assertFalse(out['IsRemediated'][0])

        self.assertAlmostEqual(out['xm'][0], 10.0, places=5)
        self.assertAlmostEqual(out['zm'][0], 0.0, places=5)
        self.assertAlmostEqual(out['sx'][0], 1.41421, places=4) # sqrt(2)
        self.assertAlmostEqual(out['sz'][0], 1.41421, places=4) # sqrt(2)

        # Check Pc value is reasonable (between 0 and 1)
        self.assertTrue(0 <= Pc[0] <= 1)

        # Miss distance is 10 km. Sigma is ~1.4 km. 10/1.4 = 7 sigma.
        # Pc should be small but not negligible.
        self.assertLess(Pc[0], 1e-3)

    def test_hbr_array(self):
        r1 = np.zeros((2, 3))
        v1 = np.zeros((2, 3))
        cov1 = np.zeros((2, 9))
        r2 = np.zeros((2, 3))
        v2 = np.zeros((2, 3))
        cov2 = np.zeros((2, 9))
        HBR = np.array([5.0, 6.0])

        # Just check it runs
        # It will warn about zero relative velocity
        with self.assertWarns(UserWarning):
            Pc, _ = PcCircle(r1, v1, cov1, r2, v2, cov2, HBR, params={'WarningLevel': 1})

        # Pc should be NaN for zero velocity
        self.assertTrue(np.isnan(Pc[0]))
        self.assertTrue(np.isnan(Pc[1]))

    def test_infinite_hbr(self):
        r1 = np.array([[1000.0, 0.0, 0.0]])
        v1 = np.array([[0.0, 7.0, 0.0]])
        cov1 = np.eye(3).reshape(1, 9)
        r2 = np.array([[1010.0, 0.0, 0.0]])
        v2 = np.array([[0.0, 7.1, 0.0]])
        cov2 = np.eye(3).reshape(1, 9)

        HBR = np.inf
        Pc, _ = PcCircle(r1, v1, cov1, r2, v2, cov2, HBR)
        self.assertEqual(Pc[0], 1.0)

    def test_zero_hbr(self):
        r1 = np.array([[1000.0, 0.0, 0.0]])
        v1 = np.array([[0.0, 7.0, 0.0]])
        cov1 = np.eye(3).reshape(1, 9)
        r2 = np.array([[1010.0, 0.0, 0.0]])
        v2 = np.array([[0.0, 7.1, 0.0]])
        cov2 = np.eye(3).reshape(1, 9)

        HBR = 0.0
        Pc, _ = PcCircle(r1, v1, cov1, r2, v2, cov2, HBR)
        self.assertEqual(Pc[0], 0.0)

    def test_estimation_mode_1(self):
        # Force adaptive integration
        r1 = np.array([[1000.0, 0.0, 0.0]])
        v1 = np.array([[0.0, 7.0, 0.0]])
        cov1 = np.eye(3).reshape(1, 9)
        r2 = np.array([[1000.1, 0.0, 0.0]]) # Close encounter
        v2 = np.array([[0.0, 7.1, 0.0]])
        cov2 = np.eye(3).reshape(1, 9)

        HBR = 1.0

        Pc, _ = PcCircle(r1, v1, cov1, r2, v2, cov2, HBR, params={'EstimationMode': 1})

        # Should be relatively high
        self.assertTrue(Pc[0] > 0.01)

    def test_estimation_mode_0(self):
        # Analytical approximation
        r1 = np.array([[1000.0, 0.0, 0.0]])
        v1 = np.array([[0.0, 7.0, 0.0]])
        cov1 = np.eye(3).reshape(1, 9)
        r2 = np.array([[1000.1, 0.0, 0.0]])
        v2 = np.array([[0.0, 7.1, 0.0]])
        cov2 = np.eye(3).reshape(1, 9)

        HBR = 1.0

        Pc, _ = PcCircle(r1, v1, cov1, r2, v2, cov2, HBR, params={'EstimationMode': 0})

        self.assertTrue(Pc[0] > 0.01)

if __name__ == '__main__':
    unittest.main()
