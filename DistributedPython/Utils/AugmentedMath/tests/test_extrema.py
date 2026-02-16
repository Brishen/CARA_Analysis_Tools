import unittest
import numpy as np
from DistributedPython.Utils.AugmentedMath.extrema import extrema

class TestExtrema(unittest.TestCase):
    def test_simple_peak(self):
        # x = [0, 1, 0]. Max at 1 (internal). Min at 0 (ends).
        x = np.array([0, 1, 0])
        xmax, imax, xmin, imin = extrema(x)
        np.testing.assert_array_equal(xmax, [1])
        np.testing.assert_array_equal(imax, [1])
        np.testing.assert_array_equal(xmin, [0, 0])
        np.testing.assert_array_equal(imin, [0, 2])

    def test_simple_peak_no_endpoints(self):
        # x = [0, 1, 0]. Max at 1. No min (since min are at endpoints).
        x = np.array([0, 1, 0])
        xmax, imax, xmin, imin = extrema(x, include_endpoints=False)
        np.testing.assert_array_equal(xmax, [1])
        np.testing.assert_array_equal(imax, [1])
        self.assertEqual(len(xmin), 0)

    def test_flat_peak(self):
        x = np.array([0, 1, 1, 1, 0])
        xmax, imax, xmin, imin = extrema(x)
        np.testing.assert_array_equal(xmax, [1])
        # Middle index is 2
        np.testing.assert_array_equal(imax, [2])
        np.testing.assert_array_equal(xmin, [0, 0])
        np.testing.assert_array_equal(imin, [0, 4])

    def test_endpoints(self):
        x = np.array([2, 1, 2])
        xmax, imax, xmin, imin = extrema(x, include_endpoints=True)
        # Maxima at ends, Minima in middle
        np.testing.assert_array_equal(xmin, [1])
        np.testing.assert_array_equal(imin, [1])
        np.testing.assert_array_equal(xmax, [2, 2])
        np.testing.assert_array_equal(imax, [0, 2])

    def test_no_endpoints(self):
        x = np.array([2, 1, 2])
        xmax, imax, xmin, imin = extrema(x, include_endpoints=False)
        np.testing.assert_array_equal(xmin, [1])
        np.testing.assert_array_equal(imin, [1])
        self.assertEqual(len(xmax), 0)

    def test_nan(self):
        x = np.array([0, np.nan, 1, 0])
        xmax, imax, xmin, imin = extrema(x)
        # Should ignore nan. effective sequence 0, 1, 0. indices 0, 2, 3.
        # Peak at 1 (index 2).
        np.testing.assert_array_equal(xmax, [1])
        np.testing.assert_array_equal(imax, [2])

    def test_flat_peak_at_end(self):
        x = np.array([0, 1, 1])
        xmax, imax, xmin, imin = extrema(x, include_endpoints=True)
        # Max at end. Flat peak.
        np.testing.assert_array_equal(xmax, [1])
        np.testing.assert_array_equal(imax, [2])
        np.testing.assert_array_equal(xmin, [0])
        np.testing.assert_array_equal(imin, [0])
