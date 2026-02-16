import unittest
import numpy as np
import math
from DistributedPython.ProbabilityOfCollision.Utils import getLebedevSphere

class TestGetLebedevSphere(unittest.TestCase):

    def test_degree_6(self):
        leb = getLebedevSphere(6)
        self.assertEqual(len(leb.x), 6)
        self.assertEqual(len(leb.y), 6)
        self.assertEqual(len(leb.z), 6)
        self.assertEqual(len(leb.w), 6)

        # Weights should sum to 4*pi
        self.assertAlmostEqual(np.sum(leb.w), 4 * math.pi, places=10)

        # Check orthogonality / integration of constant 1
        # Integral(1 dOmega) = 4*pi
        integral = np.sum(leb.w)
        self.assertAlmostEqual(integral, 4 * math.pi, places=10)

    def test_degree_14(self):
        leb = getLebedevSphere(14)
        self.assertEqual(len(leb.x), 14)
        self.assertAlmostEqual(np.sum(leb.w), 4 * math.pi, places=10)

    def test_degree_5810(self):
        leb = getLebedevSphere(5810)
        self.assertEqual(len(leb.x), 5810)
        self.assertAlmostEqual(np.sum(leb.w), 4 * math.pi, places=10)

    def test_invalid_degree(self):
        with self.assertRaises(ValueError):
            getLebedevSphere(9999)

    def test_integration_accuracy(self):
        # Integral(x^2 + y^2 + z^2) over unit sphere
        # x^2 + y^2 + z^2 = 1 on unit sphere.
        # So integral is Integral(1 dOmega) = 4*pi
        leb = getLebedevSphere(590)
        f = leb.x**2 + leb.y**2 + leb.z**2
        integral = np.sum(f * leb.w)
        self.assertAlmostEqual(integral, 4 * math.pi, places=10)

        # Integral(x^2) = 4*pi/3
        f = leb.x**2
        integral = np.sum(f * leb.w)
        self.assertAlmostEqual(integral, 4 * math.pi / 3, places=10)

if __name__ == '__main__':
    unittest.main()
