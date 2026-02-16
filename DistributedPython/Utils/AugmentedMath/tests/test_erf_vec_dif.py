import unittest
import numpy as np
from scipy.special import erf
from DistributedPython.Utils.AugmentedMath.erf_vec_dif import erf_vec_dif

class TestErfVecDif(unittest.TestCase):
    def test_basic_values(self):
        a = np.array([0.1, 0.5, 1.0])
        b = np.array([0.2, 0.6, 1.1])
        expected = erf(a) - erf(b)
        result = erf_vec_dif(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-15, atol=1e-15)

    def test_large_positive(self):
        # Case where erf(a) and erf(b) are both 1.0 in standard double precision
        a = np.array([5.0, 6.0])
        b = np.array([5.1, 6.1])

        # Standard calculation might lose precision or return 0
        std_calc = erf(a) - erf(b)

        # erf_vec_dif uses erfc, which retains precision
        result = erf_vec_dif(a, b)

        # Check that result is non-zero (since inputs are different)
        self.assertTrue(np.all(result != 0))

        # Check sign: erf is increasing, so if a < b, erf(a) < erf(b), so result < 0.
        self.assertTrue(np.all(result < 0))

    def test_large_negative(self):
        # Case where erf(a) and erf(b) are both -1.0
        a = np.array([-5.0, -6.0])
        b = np.array([-5.1, -6.1])

        result = erf_vec_dif(a, b)

        # Check that result is non-zero
        self.assertTrue(np.all(result != 0))

        # a > b (less negative), so erf(a) > erf(b), result > 0
        self.assertTrue(np.all(result > 0))

    def test_broadcasting(self):
        a = np.array([0.1, 0.5])
        b = 0.2
        expected = erf(a) - erf(b)
        result = erf_vec_dif(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-15, atol=1e-15)

    def test_scalars(self):
        a = 0.5
        b = 0.2
        expected = erf(a) - erf(b)
        result = erf_vec_dif(a, b)
        self.assertAlmostEqual(result, expected, places=15)

if __name__ == '__main__':
    unittest.main()
