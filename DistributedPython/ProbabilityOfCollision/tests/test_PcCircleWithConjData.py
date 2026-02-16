import unittest
import numpy as np
from DistributedPython.ProbabilityOfCollision.PcCircleWithConjData import PcCircleWithConjData

class TestPcCircleWithConjData(unittest.TestCase):
    def test_basic_metrics(self):
        # Create some dummy data
        r1 = np.array([[1000.0, 0.0, 0.0]])
        v1 = np.array([[0.0, 7.0, 0.0]])
        cov1 = np.eye(3).reshape(1, 9)

        r2 = np.array([[1010.0, 0.0, 0.0]])
        v2 = np.array([[0.0, 7.1, 0.0]])
        cov2 = np.eye(3).reshape(1, 9)

        HBR = 5.0

        Pc, out = PcCircleWithConjData(r1, v1, cov1, r2, v2, cov2, HBR)

        self.assertEqual(len(Pc), 1)
        self.assertIn('MissDistance', out)
        self.assertIn('SemiMajorAxis', out)
        self.assertIn('SemiMinorAxis', out)
        self.assertIn('ClockAngle', out)
        self.assertIn('x1Sigma', out)
        self.assertIn('RadialSigma', out)
        self.assertIn('InTrackSigma', out)
        self.assertIn('CrossTrackSigma', out)
        self.assertIn('RelativePhaseAngle', out)
        self.assertIn('CondNumPrimary', out)

        # Check values
        self.assertAlmostEqual(out['MissDistance'][0], 10.0, places=5)
        self.assertAlmostEqual(out['SemiMajorAxis'][0], 1.41421, places=4)
        self.assertAlmostEqual(out['SemiMinorAxis'][0], 1.41421, places=4)

        # Sigma values checks
        # CovComb is 2*I.
        # RIC frame: R along X, C along Z, I along Y?
        # r1 = [1000, 0, 0]. Rhat = [1, 0, 0].
        # v1 = [0, 7, 0].
        # h1 = r1 x v1 = [0, 0, 7000]. Chat = [0, 0, 1].
        # Ihat = Chat x Rhat = [0, 0, 1] x [1, 0, 0] = [0, 1, 0].
        # So RIC is aligned with ECI.
        # CovCombRIC should be diag([2, 2, 2]).
        # RadialSigma = sqrt(2) * 1000 approx 1414.21

        self.assertAlmostEqual(out['RadialSigma'][0], np.sqrt(2)*1000, places=3)
        self.assertAlmostEqual(out['InTrackSigma'][0], np.sqrt(2)*1000, places=3)
        self.assertAlmostEqual(out['CrossTrackSigma'][0], np.sqrt(2)*1000, places=3)

        # Condition numbers
        # C1 = I, cond = 1.
        self.assertAlmostEqual(out['CondNumPrimary'][0], 1.0, places=5)
        self.assertAlmostEqual(out['CondNumCombined'][0], 1.0, places=5)

    def test_vectorized(self):
        r1 = np.array([[1000.0, 0.0, 0.0], [2000.0, 0.0, 0.0]])
        v1 = np.array([[0.0, 7.0, 0.0], [0.0, 7.0, 0.0]])
        cov1 = np.tile(np.eye(3).reshape(1, 9), (2, 1))
        r2 = np.array([[1010.0, 0.0, 0.0], [2010.0, 0.0, 0.0]])
        v2 = np.array([[0.0, 7.1, 0.0], [0.0, 7.1, 0.0]])
        cov2 = np.tile(np.eye(3).reshape(1, 9), (2, 1))
        HBR = 5.0

        Pc, out = PcCircleWithConjData(r1, v1, cov1, r2, v2, cov2, HBR)

        self.assertEqual(len(Pc), 2)
        self.assertEqual(len(out['MissDistance']), 2)
        self.assertAlmostEqual(out['MissDistance'][1], 10.0, places=5)

if __name__ == '__main__':
    unittest.main()
