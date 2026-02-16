import unittest
import numpy as np
from DistributedPython.ProbabilityOfCollision.Utils.get_covXcorr_parameters import get_covXcorr_parameters

class TestGetCovXcorrParameters(unittest.TestCase):

    def test_missing_params(self):
        covXcorr, sigp, Gvecp, sigs, Gvecs = get_covXcorr_parameters(None)
        self.assertFalse(covXcorr)
        self.assertIsNone(sigp)
        self.assertIsNone(Gvecp)
        self.assertIsNone(sigs)
        self.assertIsNone(Gvecs)

        covXcorr, sigp, Gvecp, sigs, Gvecs = get_covXcorr_parameters({})
        self.assertFalse(covXcorr)

    def test_valid_params(self):
        params = {
            'covXcorr': {
                'sigp': 0.1,
                'sigs': 0.2,
                'Gvecp': np.array([[1, 2, 3, 4, 5, 6]]),
                'Gvecs': np.array([[6, 5, 4, 3, 2, 1]])
            }
        }
        covXcorr, sigp, Gvecp, sigs, Gvecs = get_covXcorr_parameters(params)
        self.assertTrue(covXcorr)
        self.assertEqual(sigp, 0.1)
        self.assertEqual(sigs, 0.2)
        np.testing.assert_array_equal(Gvecp, np.array([[1, 2, 3, 4, 5, 6]]))
        np.testing.assert_array_equal(Gvecs, np.array([[6, 5, 4, 3, 2, 1]]))

    def test_missing_fields(self):
        params = {
            'covXcorr': {
                'sigp': 0.1,
                # Missing others
            }
        }
        covXcorr, sigp, Gvecp, sigs, Gvecs = get_covXcorr_parameters(params)
        self.assertFalse(covXcorr)
        self.assertEqual(sigp, 0.1)
        self.assertIsNone(sigs)

    def test_incorrect_dimensions(self):
        # sigp not scalar
        params = {
            'covXcorr': {
                'sigp': [0.1],
                'sigs': 0.2,
                'Gvecp': np.array([[1, 2, 3, 4, 5, 6]]),
                'Gvecs': np.array([[6, 5, 4, 3, 2, 1]])
            }
        }
        with self.assertRaisesRegex(ValueError, 'Incorrect DCP value dimensions'):
             get_covXcorr_parameters(params)

        # Gvecp wrong shape (1D instead of 2D 1x6)
        params['covXcorr']['sigp'] = 0.1
        params['covXcorr']['Gvecp'] = np.array([1, 2, 3, 4, 5, 6])
        with self.assertRaisesRegex(ValueError, 'Incorrect DCP value dimensions'):
             get_covXcorr_parameters(params)

        # Gvecp wrong shape (2x3)
        params['covXcorr']['Gvecp'] = np.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaisesRegex(ValueError, 'Incorrect DCP value dimensions'):
             get_covXcorr_parameters(params)

    def test_invalid_values(self):
        # Negative sigma
        params = {
            'covXcorr': {
                'sigp': -0.1,
                'sigs': 0.2,
                'Gvecp': np.array([[1, 2, 3, 4, 5, 6]]),
                'Gvecs': np.array([[6, 5, 4, 3, 2, 1]])
            }
        }
        with self.assertRaisesRegex(ValueError, 'Invalid DCP sigma value'):
             get_covXcorr_parameters(params)

        # NaN sigma
        params['covXcorr']['sigp'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Invalid DCP sigma value'):
             get_covXcorr_parameters(params)

        # NaN in Gvecp
        params['covXcorr']['sigp'] = 0.1
        params['covXcorr']['Gvecp'] = np.array([[1, np.nan, 3, 4, 5, 6]])
        with self.assertRaisesRegex(ValueError, 'Invalid DCP sensitivity vector value'):
             get_covXcorr_parameters(params)

if __name__ == '__main__':
    unittest.main()
