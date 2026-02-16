import datetime
import warnings
import numpy as np
from DistributedPython.ProbabilityOfCollision.Utils.set_default_param import set_default_param

def default_params_Pc_SDMC(params=None):
    """
    Add and/or set the defaults for the parameters used by the function Pc_SDMC.

    Args:
        params (dict, optional): Empty or partially populated dictionary containing the
                                 parameters used by the Pc_SDMC function.

    Returns:
        dict: Fully-populated dictionary containing parameters used by the function Pc_SDMC.
    """
    if params is None:
        params = {}

    # Default trajectory mode
    #   0 = 2-body motion
    #   1 = full rectilinear motion
    #   2 = rectilinear motion (position deviations only)
    params = set_default_param(params, 'trajectory_mode', 0)
    if params['trajectory_mode'] not in [0, 1, 2]:
        raise ValueError('Supplied trajectory_mode must be 0, 1, or 2')

    # Target 95% confidence Pc estimation accuracy
    # (0.1 corresponds to ~400 hits; 0.2 corresponds to ~100 hits)
    params = set_default_param(params, 'Target95pctPcAccuracy', 0.1)

    # Max number of trials allowed (to prevent needlessly long runs)
    params = set_default_param(params, 'max_num_trials', 3.7e7)

    # Default number of trials (when a Pc cannot be calculated analytically)
    params = set_default_param(params, 'default_num_trials', 3.7e7)

    # Outputs from SDMC library
    #  ''       = no outputs
    #  'STDOUT' = printed to standard out
    #  fileName = printed to a file name (fileName cannot exceed 255 characters)
    params = set_default_param(params, 'sdmc_list_file', '')
    if not isinstance(params['sdmc_list_file'], str) or len(params['sdmc_list_file']) > 255:
        raise ValueError('Supplied sdmc_list_file must be a string less than or equal to 255 characters')

    # Process SDMC with parallel processes
    # Python multiprocessing is generally available, so we default to True unless explicitly disabled.
    params = set_default_param(params, 'use_parallel', True)

    # Default seed
    params = set_default_param(params, 'seed', -1)

    seed = params['seed']
    if (seed >= 0 or np.isinf(seed) or np.isnan(seed) or
        int(seed) != seed or seed < -2147483648):
         raise ValueError('Supplied seed must be a negative integer greater than -2,147,483,649')

    # Default confidence level for Pc uncertainty
    params = set_default_param(params, 'conf_level', 0.95)
    if params['conf_level'] <= 0 or params['conf_level'] >= 1:
        raise ValueError('Supplied conf_level must be a number between 0 and 1 (not inclusive)')

    # Conjunction close approach distribution control
    params = set_default_param(params, 'generate_ca_dist_plot', False)

    # Check the number of trials parameter
    params = set_default_param(params, 'num_trials', None)
    if params['num_trials'] is not None:
        num_trials = params['num_trials']
        if (np.isinf(num_trials) or np.isnan(num_trials) or
            num_trials <= 0 or int(num_trials) != num_trials):
            raise ValueError('Supplied num_trials must be a positive integer value')
        elif num_trials > 1e9:
            warnings.warn(f"Supplied num_trials ({num_trials}) is above the recommended maximum of 1e9\n"
                          "It is expected that this process will run for a long time")

    # Set the number of workers
    params = set_default_param(params, 'num_workers', None)
    if params['num_workers'] is not None:
        num_workers = params['num_workers']
        if (np.isinf(num_workers) or np.isnan(num_workers) or
            num_workers <= 0 or int(num_workers) != num_workers):
            raise ValueError('Supplied num_workers must be a positive integer value')

    # Check the span days parameter
    params = set_default_param(params, 'span_days', None)

    # Check tmid
    params = set_default_param(params, 'tmid', None)

    # InputPc is used to determine the number of trials to run
    params = set_default_param(params, 'InputPc', None)

    # Initialize covariance cross correlation correction indicator flag
    params = set_default_param(params, 'apply_covXcorr_corrections', True)

    # Check primary and secondary object IDs
    params = set_default_param(params, 'pri_objectid', 1)
    params = set_default_param(params, 'sec_objectid', 2)

    # Check TCA, pri epoch, and sec epoch.
    default_date = datetime.datetime(1990, 1, 1)
    params = set_default_param(params, 'TCA', default_date)
    params = set_default_param(params, 'pri_epoch', params['TCA'])
    params = set_default_param(params, 'sec_epoch', params['TCA'])

    if params['TCA'] != params['pri_epoch'] or params['TCA'] != params['sec_epoch']:
        warnings.warn("Supplied TCA, pri_epoch, and sec_epoch do not match.\n"
                      "This is not a recommended run configuration for SDMC.")

    # Set the default retrograde orbit reorientation mode
    #  0 => No retrograde orbit adjustment
    #  1 => If either orbits is retrograde, try reorienting the reference frame axes (recommended)
    #  2 => Always try reorienting the ref. frame axes (testing mode)
    #  3 => Reorient axes to force primary to be retrograde (testing mode)
    params = set_default_param(params, 'RetrogradeReorientation', 1)

    # Validate target accuracy parameter
    tgt_acc = params['Target95pctPcAccuracy']
    # Check if scalar
    if np.ndim(tgt_acc) == 0:
        if tgt_acc <= 0 or tgt_acc >= 1:
            warnings.warn('Invalid Target95pctPcAccuracy parameter; setting to 10%')
            params['Target95pctPcAccuracy'] = 0.1
    else:
        # Check if Nx2 array
        tgt_acc = np.array(tgt_acc)
        if tgt_acc.ndim != 2 or tgt_acc.shape[1] != 2 or tgt_acc.shape[0] < 1:
             warnings.warn('Invalid Target95pctPcAccuracy table; setting to 10%')
             params['Target95pctPcAccuracy'] = 0.1

    # Default warning level
    params = set_default_param(params, 'warning_level', 1)

    # Default verbosity
    params = set_default_param(params, 'verbose', False)

    return params
