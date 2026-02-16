import unittest
import numpy as np
from DistributedPython.Utils.AugmentedMath.refine_bounded_extrema import refine_bounded_extrema

class TestRefineBoundedExtrema(unittest.TestCase):
    def test_simple_parabola(self):
        # y = -(x-2)^2 + 1 = -x^2 + 4x - 3. Max at x=2, y=1.
        fun = lambda x: -(x-2)**2 + 1
        xmnma, ymnma, xmxma, ymxma, converged, nbisect, x, y, imnma, imxma = \
            refine_bounded_extrema(fun, 0, 4, Ninitial=5, verbose=0, extrema_types=2)

        self.assertTrue(converged)
        self.assertEqual(len(xmxma), 1)
        self.assertAlmostEqual(xmxma[0], 2.0, places=4)
        self.assertAlmostEqual(ymxma[0], 1.0, places=4)
        self.assertEqual(len(xmnma), 0)

    def test_mode_2(self):
        # y = (x-2)^2. Min at x=2, y=0.
        x = np.linspace(0, 4, 5) # 0, 1, 2, 3, 4
        y = (x-2)**2 # 4, 1, 0, 1, 4
        fun = lambda x: (x-2)**2

        xmnma, ymnma, xmxma, ymxma, converged, nbisect, x_out, y_out, imnma, imxma = \
            refine_bounded_extrema(fun, x, y, Ninitial=None, verbose=0, extrema_types=1)

        self.assertTrue(converged)
        self.assertEqual(len(xmnma), 1)
        self.assertAlmostEqual(xmnma[0], 2.0, places=4)
        self.assertAlmostEqual(ymnma[0], 0.0, places=4)

    def test_no_extrema(self):
        fun = lambda x: x
        xmnma, ymnma, xmxma, ymxma, converged, nbisect, x, y, imnma, imxma = \
            refine_bounded_extrema(fun, 0, 1, Ninitial=5, verbose=0)

        # Linear function has no interior extrema
        self.assertEqual(len(xmnma), 0)
        self.assertEqual(len(xmxma), 0)
