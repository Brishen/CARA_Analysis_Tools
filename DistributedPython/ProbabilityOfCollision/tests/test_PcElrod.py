import unittest
import numpy as np
import warnings
from DistributedPython.ProbabilityOfCollision.PcElrod import PcElrod

class TestPcElrod(unittest.TestCase):
    def test_basic_validation(self):
        # Create some dummy data
        r1 = np.array([[1000.0, 0.0, 0.0]])
        v1 = np.array([[0.0, 7.0, 0.0]])
        cov1 = np.eye(3).reshape(1, 9)

        r2 = np.array([[1010.0, 0.0, 0.0]])
        v2 = np.array([[0.0, 7.1, 0.0]])
        cov2 = np.eye(3).reshape(1, 9)

        HBR = 5.0

        Pc, Arem, IsPosDef, IsRemediated = PcElrod(r1, v1, cov1, r2, v2, cov2, HBR)

        self.assertEqual(len(Pc), 1)
        self.assertTrue(IsPosDef[0])
        self.assertFalse(IsRemediated[0])

        # Check Pc value is reasonable (between 0 and 1)
        self.assertTrue(0 <= Pc[0] <= 1)

        # Miss distance is 10 km. Sigma is ~1.4 km. 10/1.4 = 7 sigma.
        # Pc should be small but not negligible.
        self.assertLess(Pc[0], 1e-3)

    def test_invalid_dimensions(self):
        r1 = np.array([[1000.0, 0.0, 0.0]])
        v1 = np.array([[0.0, 7.0, 0.0]])
        cov1 = np.eye(3).reshape(1, 9)
        r2 = np.array([[1010.0, 0.0, 0.0], [1020.0, 0.0, 0.0]]) # 2 positions
        v2 = np.array([[0.0, 7.1, 0.0]])
        cov2 = np.eye(3).reshape(1, 9)
        HBR = 5.0

        with self.assertRaisesRegex(ValueError, "UnequalPositionCount"):
             PcElrod(r1, v1, cov1, r2, v2, cov2, HBR)

    def test_hbr_array(self):
        r1 = np.zeros((2, 3))
        v1 = np.zeros((2, 3))
        cov1 = np.zeros((2, 9))
        r2 = np.zeros((2, 3))
        v2 = np.zeros((2, 3))
        cov2 = np.zeros((2, 9))
        HBR = np.array([5.0, 6.0])

        # Just check it runs
        # It will likely produce NaN or error due to zero relative velocity/position
        # But should not crash due to dimensions.
        Pc, _, _, _ = PcElrod(r1, v1, cov1, r2, v2, cov2, HBR)

        self.assertTrue(np.isnan(Pc[0]))
        self.assertTrue(np.isnan(Pc[1]))

    def test_warning_level(self):
         # Trigger negative HBR warning
        r1 = np.array([[1000.0, 0.0, 0.0]])
        v1 = np.array([[0.0, 7.0, 0.0]])
        cov1 = np.eye(3).reshape(1, 9)
        r2 = np.array([[1010.0, 0.0, 0.0]])
        v2 = np.array([[0.0, 7.1, 0.0]])
        cov2 = np.eye(3).reshape(1, 9)

        HBR = -5.0

        with self.assertWarns(UserWarning):
            PcElrod(r1, v1, cov1, r2, v2, cov2, HBR, WarningLevel=1)

if __name__ == '__main__':
    unittest.main()
