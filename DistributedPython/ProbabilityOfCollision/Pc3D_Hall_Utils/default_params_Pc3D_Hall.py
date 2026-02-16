import numpy as np
import warnings
from DistributedPython.ProbabilityOfCollision.Utils.getLebedevSphere import getLebedevSphere
from DistributedPython.ProbabilityOfCollision.Utils.set_default_param import set_default_param

# Persistent variables emulation
_sph_Lebedev = None
_deg_Lebedev = None

def default_params_Pc3D_Hall(params=None, Lebedev_warning=True):
    """
    Add and/or set the defaults for the parameters used by the function Pc3D_Hall.

    Parameters
    ----------
    params : dict, optional
        Empty or partially populated dictionary containing the parameters.
    Lebedev_warning : bool, optional
        Flag to issue a warning when Lebedev unit-sphere quadrature points are being recalculated.

    Returns
    -------
    params : dict
        Fully-populated dictionary containing parameters used by the function Pc3D_Hall.
    """
    global _sph_Lebedev, _deg_Lebedev

    if params is None:
        params = {}

    # Default gamma factor (see C12b).
    if params.get('gamma') is None:
        params['gamma'] = 1e-16

    # Ensure that the input value for gamma is single-valued
    gamma = np.atleast_1d(params['gamma'])
    Ngamma = gamma.size
    if Ngamma > 1:
        warnings.warn('Multiple gamma values supplied; using minimum value.')
        params['gamma'] = np.min(gamma)

    # Ensure that the input value for C12b's gamma is sensible
    if params['gamma'] >= 1 or params['gamma'] <= 0:
        raise ValueError(f"Supplied gamma = {params['gamma']} is out-of-range. RECOMMENDATION: 1e-16 <= gamma <= 1e-6.")
    elif params['gamma'] > 1e-6:
        warnings.warn(f"Supplied gamma = {params['gamma']} is relatively large. RECOMMENDATION: 1e-16 <= gamma <= 1e-6.")

    # Default number of ephemeris points to use
    params = set_default_param(params, 'Neph', 101)

    # Default number of peak PDF overlap position iterations
    params = set_default_param(params, 'POPmaxiter', 100)

    # Default expansion factor for conjunction time bounds
    if 'Texpand' not in params:
        params['Texpand'] = None

    Texpand = params['Texpand']
    if Texpand is not None:
        Texpand = np.atleast_1d(Texpand)
        if Texpand.size == 1:
            if np.isnan(Texpand[0]) or np.isinf(Texpand[0]) or Texpand[0] <= 0:
                raise ValueError('Invalid Texpand parameter')
            params['Texpand'] = Texpand[0] # Store as scalar if single value
        elif Texpand.size == 2:
            if np.any(np.isnan(Texpand)) or np.any(np.isinf(Texpand)) or Texpand[0] >= Texpand[1]:
                raise ValueError('Invalid Texpand parameters')
            params['Texpand'] = Texpand # Store as array/list
        else:
             raise ValueError('Invalid Texpand parameter size')

    # Default limits for conjunction segment duration (used for Texpand = [])
    params = set_default_param(params, 'Tmin_limit', None)
    params = set_default_param(params, 'Tmax_limit', None)

    # Check for invalid limits
    if params['Texpand'] is None:
        Tmin_limit = params['Tmin_limit']
        Tmax_limit = params['Tmax_limit']
        isempty_Tmin_limit = Tmin_limit is None
        isempty_Tmax_limit = Tmax_limit is None

        bad_limits = False
        bad_Tmin_limit = False
        bad_Tmax_limit = False

        if isempty_Tmin_limit and isempty_Tmax_limit:
            bad_limits = False
        else:
            if not isempty_Tmin_limit:
                bad_Tmin_limit = np.isnan(Tmin_limit) or np.isinf(Tmin_limit)
            if not isempty_Tmax_limit:
                bad_Tmax_limit = np.isnan(Tmax_limit) or np.isinf(Tmax_limit)

            if bad_Tmin_limit or bad_Tmax_limit:
                bad_limits = True
            else:
                if not isempty_Tmin_limit and not isempty_Tmax_limit:
                    bad_limits = Tmin_limit >= Tmax_limit
                else:
                    bad_limits = False

        if bad_limits:
            raise ValueError('Invalid Tmin_limit and/or Tmax_limit parameters')

    # Default limits for initial conjunction duration (used for Texpand = [])
    params = set_default_param(params, 'Tmin_initial', None)
    params = set_default_param(params, 'Tmax_initial', None)

    if params['Texpand'] is None:
        Tmin_initial = params['Tmin_initial']
        Tmax_initial = params['Tmax_initial']
        isempty_Tmin_initial = Tmin_initial is None
        isempty_Tmax_initial = Tmax_initial is None

        bad_initials = False
        if isempty_Tmin_initial and isempty_Tmax_initial:
            bad_initials = False
        elif not isempty_Tmin_initial and not isempty_Tmax_initial:
            if Tmin_initial == -np.inf and Tmax_initial == np.inf:
                bad_initials = False
            elif np.isnan(Tmin_initial) or np.isinf(Tmin_initial) or \
                 np.isnan(Tmax_initial) or np.isinf(Tmax_initial) or \
                 (Tmin_initial >= Tmax_initial):
                bad_initials = True
            else:
                bad_initials = False
        else:
            bad_initials = True

        if bad_initials:
            raise ValueError('Invalid Tmin_initial and/or Tmax_initial parameters')

        params = set_default_param(params, 'AccelerateRefinement', False)

    # Default for flag to restrict conjunction durations
    params = set_default_param(params, 'Torblimit', True)

    # Default for flag to use NPD-remediated equinoctial covariances
    params = set_default_param(params, 'remediate_NPD_TCA_eq_covariances', False)

    # Initialize covariance cross correlation correction indicator flag
    params = set_default_param(params, 'apply_covXcorr_corrections', True)

    # Set up the unit-sphere integration parameters
    params = set_default_param(params, 'use_Lebedev', True)
    params = set_default_param(params, 'deg_Lebedev', 5810)
    params = set_default_param(params, 'slow_method', False)
    params = set_default_param(params, 'AbsTol', 0)
    params = set_default_param(params, 'RelTol', 1e-9)
    params = set_default_param(params, 'MaxFunEvals', 10000)

    # Default eigenvalue clipping factor
    params = set_default_param(params, 'Fclip', 1e-4)

    # Default tiny Pc value
    params = set_default_param(params, 'Pc_tiny', 1e-300)

    # Default GM value assumes meters for length units
    params = set_default_param(params, 'GM', 3.986004418e14)

    # Set the default retrograde orbit reorientation mode
    params = set_default_param(params, 'RetrogradeReorientation', 1)

    # Default verbosity
    params = set_default_param(params, 'verbose', 0)

    # Set up Lebedev structure for unit-sphere integrations
    if params['use_Lebedev']:
        # Check if Lebedev weights and vectors are defined and correct
        wgt_Lebedev = params.get('wgt_Lebedev')
        vec_Lebedev = params.get('vec_Lebedev')

        if wgt_Lebedev is None or vec_Lebedev is None:
            calc_Leb = True
        else:
            wgt_Lebedev = np.asarray(wgt_Lebedev)
            if wgt_Lebedev.size != params['deg_Lebedev']:
                calc_Leb = True
            else:
                calc_Leb = False

        if calc_Leb:
            params = set_default_param(params, 'suppress_Lebedev_warning', False)

            if _sph_Lebedev is None:
                _sph_Lebedev = getLebedevSphere(params['deg_Lebedev'])
                _deg_Lebedev = params['deg_Lebedev']
            elif _deg_Lebedev != params['deg_Lebedev']:
                _sph_Lebedev = getLebedevSphere(params['deg_Lebedev'])
                _deg_Lebedev = params['deg_Lebedev']
                if Lebedev_warning and not params['suppress_Lebedev_warning']:
                    warnings.warn('Lebedev quadrature points recalculated; needless repetition can slow execution.')

            # Copy Lebedev vector and weights from persistent variable
            # Python struct return from getLebedevSphere: x, y, z, w are 1D arrays
            # params.vec_Lebedev = [sph_Lebedev.x sph_Lebedev.y sph_Lebedev.z]'; -> (3, N)
            params['vec_Lebedev'] = np.vstack((_sph_Lebedev.x, _sph_Lebedev.y, _sph_Lebedev.z))
            params['wgt_Lebedev'] = _sph_Lebedev.w

    return params
