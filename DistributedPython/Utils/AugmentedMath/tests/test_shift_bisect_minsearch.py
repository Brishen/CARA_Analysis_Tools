import unittest
import numpy as np
from DistributedPython.Utils.AugmentedMath.shift_bisect_minsearch import shift_bisect_minsearch

class TestShiftBisectMinsearch(unittest.TestCase):

    def test_simple_parabola_min_in_center(self):
        # f(x) = (x-2)^2, min at x=2
        fun = lambda x: (x - 2.0)**2
        x0 = np.linspace(0, 4, 5) # [0, 1, 2, 3, 4]
        y0 = fun(x0) # [4, 1, 0, 1, 4]

        xmin, ymin, converged, nbisect, nshift, xbuf, ybuf = shift_bisect_minsearch(fun, x0, y0)

        self.assertTrue(converged)
        self.assertAlmostEqual(xmin, 2.0, places=4)
        self.assertAlmostEqual(ymin, 0.0, places=4)
        self.assertEqual(nshift, 0) # Should be centered

    def test_parabola_min_outside_right(self):
        # f(x) = (x-6)^2, min at x=6
        fun = lambda x: (x - 6.0)**2
        x0 = np.linspace(0, 4, 5) # [0, 1, 2, 3, 4]
        y0 = fun(x0) # [36, 25, 16, 9, 4] -> min at x=4 (right edge)

        xmin, ymin, converged, nbisect, nshift, xbuf, ybuf = shift_bisect_minsearch(fun, x0, y0)

        self.assertTrue(converged)
        self.assertAlmostEqual(xmin, 6.0, places=4)
        self.assertAlmostEqual(ymin, 0.0, places=4)
        self.assertGreater(nshift, 0)

    def test_parabola_min_outside_left(self):
        # f(x) = (x+2)^2, min at x=-2
        fun = lambda x: (x + 2.0)**2
        x0 = np.linspace(0, 4, 5) # [0, 1, 2, 3, 4]
        y0 = fun(x0) # [4, 9, 16, 25, 36] -> min at x=0 (left edge)

        xmin, ymin, converged, nbisect, nshift, xbuf, ybuf = shift_bisect_minsearch(fun, x0, y0)

        self.assertTrue(converged)
        self.assertAlmostEqual(xmin, -2.0, places=4)
        self.assertAlmostEqual(ymin, 0.0, places=4)
        self.assertGreater(nshift, 0)

    def test_buffering(self):
        fun = lambda x: x**2
        x0 = np.linspace(-2, 2, 5)
        y0 = fun(x0)

        xmin, ymin, converged, nbisect, nshift, xbuf, ybuf = shift_bisect_minsearch(fun, x0, y0)

        self.assertTrue(converged)
        self.assertIsNotNone(xbuf)
        self.assertIsNotNone(ybuf)
        self.assertEqual(len(xbuf), len(ybuf))
        self.assertTrue(len(xbuf) >= 5)

    def test_invalid_input_size(self):
        fun = lambda x: x**2
        x0 = np.array([1, 2, 3])
        y0 = fun(x0)
        with self.assertRaises(ValueError):
            shift_bisect_minsearch(fun, x0, y0)

    def test_mismatched_input(self):
        fun = lambda x: x**2
        x0 = np.array([1, 2, 3, 4, 5])
        y0 = np.array([1, 4, 9, 16])
        with self.assertRaises(ValueError):
            shift_bisect_minsearch(fun, x0, y0)

    def test_non_monotonic_input(self):
        fun = lambda x: x**2
        x0 = np.array([1, 3, 2, 4, 5])
        y0 = fun(x0)
        with self.assertRaises(ValueError):
            shift_bisect_minsearch(fun, x0, y0)

if __name__ == '__main__':
    unittest.main()
