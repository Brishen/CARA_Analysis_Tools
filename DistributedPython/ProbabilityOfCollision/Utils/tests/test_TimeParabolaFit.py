import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from DistributedPython.ProbabilityOfCollision.Utils.TimeParabolaFit import TimeParabolaFit

class TestTimeParabolaFit(unittest.TestCase):

    def test_simple_parabola(self):
        # y = x^2
        t = np.array([-1.0, 0.0, 1.0])
        F = t**2

        res = TimeParabolaFit(t, F)

        c = res['c']
        # Expected: c[0]=1, c[1]=0, c[2]=0
        self.assertTrue(np.allclose(c, [1.0, 0.0, 0.0]))
        self.assertEqual(res['rankAmat'], 3)

    def test_shifted_parabola(self):
        # y = 2(x-1)^2 + 3 = 2(x^2 - 2x + 1) + 3 = 2x^2 - 4x + 5
        t = np.array([0.0, 1.0, 2.0])
        F = 2*(t-1)**2 + 3

        res = TimeParabolaFit(t, F)

        c = res['c']
        self.assertTrue(np.allclose(c, [2.0, -4.0, 5.0]))

    def test_minimum_points(self):
        # 3 points required
        t = np.array([1.0, 2.0])
        F = np.array([1.0, 2.0])

        with self.assertRaises(ValueError):
            TimeParabolaFit(t, F)

    def test_duplicate_points(self):
        # Duplicate points should be filtered
        t = np.array([1.0, 1.0, 2.0, 3.0])
        F = np.array([1.0, 1.0, 4.0, 9.0]) # y = x^2

        res = TimeParabolaFit(t, F)

        c = res['c']
        self.assertTrue(np.allclose(c, [1.0, 0.0, 0.0]))
        # Check that tinc has only unique points
        self.assertEqual(len(res['tinc']), 3)

    def test_more_points(self):
        # More points provided, but should use minimal set (3) closest to minimum F
        t = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        F = t**2 # Min is at t=0, F=0

        # Sorted F: 0 (t=0), 1 (t=-1), 1 (t=1), 4 (t=-2), 4 (t=2)
        # Should use t=0, t=-1, t=1.

        res = TimeParabolaFit(t, F)

        tinc = res['tinc']
        # tinc order might vary but set should be same
        self.assertTrue(np.all(np.isin(tinc, [0.0, -1.0, 1.0])))
        self.assertEqual(len(tinc), 3)

        c = res['c']
        self.assertTrue(np.allclose(c, [1.0, 0.0, 0.0]))

if __name__ == '__main__':
    unittest.main()
