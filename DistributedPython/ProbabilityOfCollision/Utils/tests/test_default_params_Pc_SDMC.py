import unittest
import datetime
import numpy as np
from DistributedPython.ProbabilityOfCollision.Utils.default_params_Pc_SDMC import default_params_Pc_SDMC

class TestDefaultParamsPcSDMC(unittest.TestCase):

    def test_defaults(self):
        params = default_params_Pc_SDMC()

        self.assertEqual(params['trajectory_mode'], 0)
        self.assertEqual(params['Target95pctPcAccuracy'], 0.1)
        self.assertEqual(params['max_num_trials'], 3.7e7)
        self.assertEqual(params['default_num_trials'], 3.7e7)
        self.assertEqual(params['sdmc_list_file'], '')
        self.assertTrue(params['use_parallel'])
        self.assertEqual(params['seed'], -1)
        self.assertEqual(params['conf_level'], 0.95)
        self.assertFalse(params['generate_ca_dist_plot'])
        self.assertIsNone(params['num_trials'])
        self.assertIsNone(params['num_workers'])
        self.assertIsNone(params['span_days'])
        self.assertIsNone(params['tmid'])
        self.assertIsNone(params['InputPc'])
        self.assertTrue(params['apply_covXcorr_corrections'])
        self.assertEqual(params['pri_objectid'], 1)
        self.assertEqual(params['sec_objectid'], 2)

        default_date = datetime.datetime(1990, 1, 1)
        self.assertEqual(params['TCA'], default_date)
        self.assertEqual(params['pri_epoch'], default_date)
        self.assertEqual(params['sec_epoch'], default_date)

        self.assertEqual(params['RetrogradeReorientation'], 1)
        self.assertEqual(params['warning_level'], 1)
        self.assertFalse(params['verbose'])

    def test_overrides(self):
        input_params = {
            'trajectory_mode': 1,
            'seed': -12345,
            'Target95pctPcAccuracy': 0.05
        }
        params = default_params_Pc_SDMC(input_params)

        self.assertEqual(params['trajectory_mode'], 1)
        self.assertEqual(params['seed'], -12345)
        self.assertEqual(params['Target95pctPcAccuracy'], 0.05)
        # Check that others are default
        self.assertEqual(params['max_num_trials'], 3.7e7)

    def test_invalid_trajectory_mode(self):
        with self.assertRaisesRegex(ValueError, 'Supplied trajectory_mode must be 0, 1, or 2'):
            default_params_Pc_SDMC({'trajectory_mode': 3})

    def test_invalid_sdmc_list_file(self):
        with self.assertRaisesRegex(ValueError, 'Supplied sdmc_list_file must be a string'):
            default_params_Pc_SDMC({'sdmc_list_file': 123})

        long_string = 'a' * 256
        with self.assertRaisesRegex(ValueError, 'Supplied sdmc_list_file must be a string'):
            default_params_Pc_SDMC({'sdmc_list_file': long_string})

    def test_invalid_seed(self):
        with self.assertRaisesRegex(ValueError, 'Supplied seed must be a negative integer'):
            default_params_Pc_SDMC({'seed': 1})
        with self.assertRaisesRegex(ValueError, 'Supplied seed must be a negative integer'):
            default_params_Pc_SDMC({'seed': 0})
        with self.assertRaisesRegex(ValueError, 'Supplied seed must be a negative integer'):
            default_params_Pc_SDMC({'seed': -1.5})

    def test_invalid_conf_level(self):
        with self.assertRaisesRegex(ValueError, 'Supplied conf_level must be a number between 0 and 1'):
            default_params_Pc_SDMC({'conf_level': 1.1})
        with self.assertRaisesRegex(ValueError, 'Supplied conf_level must be a number between 0 and 1'):
            default_params_Pc_SDMC({'conf_level': 0})

    def test_invalid_num_trials(self):
        with self.assertRaisesRegex(ValueError, 'Supplied num_trials must be a positive integer'):
            default_params_Pc_SDMC({'num_trials': -1})
        with self.assertRaisesRegex(ValueError, 'Supplied num_trials must be a positive integer'):
            default_params_Pc_SDMC({'num_trials': 1.5})

        with self.assertWarnsRegex(UserWarning, 'above the recommended maximum'):
            default_params_Pc_SDMC({'num_trials': 2e9})

    def test_invalid_num_workers(self):
        with self.assertRaisesRegex(ValueError, 'Supplied num_workers must be a positive integer'):
            default_params_Pc_SDMC({'num_workers': -1})

    def test_epoch_mismatch_warning(self):
        date1 = datetime.datetime(1990, 1, 1)
        date2 = datetime.datetime(1991, 1, 1)
        with self.assertWarnsRegex(UserWarning, 'Supplied TCA, pri_epoch, and sec_epoch do not match'):
            default_params_Pc_SDMC({'TCA': date1, 'pri_epoch': date2})

    def test_invalid_target_accuracy(self):
        with self.assertWarnsRegex(UserWarning, 'Invalid Target95pctPcAccuracy parameter'):
             params = default_params_Pc_SDMC({'Target95pctPcAccuracy': 1.1})
             self.assertEqual(params['Target95pctPcAccuracy'], 0.1)

        with self.assertWarnsRegex(UserWarning, 'Invalid Target95pctPcAccuracy parameter'):
             params = default_params_Pc_SDMC({'Target95pctPcAccuracy': -0.1})
             self.assertEqual(params['Target95pctPcAccuracy'], 0.1)

    def test_valid_target_accuracy_table(self):
        # Nx2 table
        table = np.array([[1e-7, 0.1], [1e-5, 0.2]])
        params = default_params_Pc_SDMC({'Target95pctPcAccuracy': table})
        np.testing.assert_array_equal(params['Target95pctPcAccuracy'], table)

    def test_invalid_target_accuracy_table(self):
        # Invalid shape
        table = np.array([0.1, 0.2])
        with self.assertWarnsRegex(UserWarning, 'Invalid Target95pctPcAccuracy table'):
             params = default_params_Pc_SDMC({'Target95pctPcAccuracy': table})
             self.assertEqual(params['Target95pctPcAccuracy'], 0.1)

if __name__ == '__main__':
    unittest.main()
