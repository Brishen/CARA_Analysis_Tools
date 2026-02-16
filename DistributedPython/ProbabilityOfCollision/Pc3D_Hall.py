import numpy as np
import scipy.special
import scipy.integrate
from scipy.optimize import minimize_scalar
import warnings

# Utils imports
from DistributedPython.ProbabilityOfCollision.Pc3D_Hall_Utils.default_params_Pc3D_Hall import default_params_Pc3D_Hall
from DistributedPython.ProbabilityOfCollision.Pc3D_Hall_Utils.delta_r2_equin import delta_r2_equin
from DistributedPython.ProbabilityOfCollision.Pc3D_Hall_Utils.multiprod import multiprod
from DistributedPython.ProbabilityOfCollision.Pc3D_Hall_Utils.multitransp import multitransp
from DistributedPython.ProbabilityOfCollision.Utils.conj_bounds_Coppola import conj_bounds_Coppola
from DistributedPython.ProbabilityOfCollision.Utils.jacobian_E0_to_Xt import jacobian_E0_to_Xt
from DistributedPython.ProbabilityOfCollision.Utils.PeakOverlapPos import PeakOverlapPos
from DistributedPython.ProbabilityOfCollision.Utils.CovRemEigValClip import CovRemEigValClip
from DistributedPython.ProbabilityOfCollision.Utils.EquinoctialMatrices import EquinoctialMatrices
from DistributedPython.ProbabilityOfCollision.Utils.RetrogradeReorientation import RetrogradeReorientation
from DistributedPython.ProbabilityOfCollision.Utils.get_covXcorr_parameters import get_covXcorr_parameters
from DistributedPython.Utils.OrbitTransformations.orbit_period import orbit_period
from DistributedPython.Utils.AugmentedMath.cov_make_symmetric import cov_make_symmetric
from DistributedPython.Utils.AugmentedMath.extrema import extrema
from DistributedPython.Utils.AugmentedMath.refine_bounded_extrema import refine_bounded_extrema


def Pc3D_Hall(r1, v1, C1, r2, v2, C2, HBR, params=None):
    """
    Calculate the Hall (2021) approximation for the probability of collision
    between two satellites, given input states and covariances at the nominal
    time of closest approach (TCA).

    Parameters
    ----------
    r1 : array_like
        Primary object's ECI position vector (m). [3x1 or 1x3]
    v1 : array_like
        Primary object's ECI velocity vector (m/s). [3x1 or 1x3]
    C1 : array_like
        Primary object's ECI covariance matrix (m position units). [6x6]
    r2 : array_like
        Secondary object's ECI position vector (m). [3x1 or 1x3]
    v2 : array_like
        Secondary object's ECI velocity vector (m/s). [3x1 or 1x3]
    C2 : array_like
        Secondary object's ECI covariance matrix (m position units). [6x6]
    HBR : float or array_like
        Combined primary+secondary hard-body radii. [1x1, 1x2, or 2x1]
    params : dict, optional
        Auxiliary input parameter structure.

    Returns
    -------
    Pc : float
        The estimated conjunction-integrated Pc value (i.e., Nc value).
    out : dict
        Auxiliary output structure.
    """

    # Initializations and defaults
    if params is None:
        params = {}
        params['verbose'] = False

    if 'verbose' not in params:
        params['verbose'] = False

    # Set debug_instrumentation default
    if 'debug_instrumentation' not in params or params['debug_instrumentation'] is None:
        params['debug_instrumentation'] = False

    if params['debug_instrumentation']:
        if 'debug_output_file' not in params or params['debug_output_file'] is None:
            params['debug_output_file'] = 'Pc3D_Hall_debug_py.txt'
        # Clear the debug file
        try:
            with open(params['debug_output_file'], 'w'):
                pass
        except Exception as e:
            import sys
            sys.stderr.write(f"Error clearing debug file: {e}\n")

    # Set parameters defaults
    params = default_params_Pc3D_Hall(params)

    # Debug plotting flag
    if 'debug_plotting' not in params or params['debug_plotting'] is None:
        params['debug_plotting'] = 0

    refinement_debug_plotting = params['debug_plotting']
    if refinement_debug_plotting:
        params['verbose'] = True

    # Copy parameters to the output structure
    out = {'params': params}

    # Ensure primary/secondary state vectors are column vectors (3x1)
    # In Python/Numpy, usually 1D arrays of shape (3,) are preferred for vectors,
    # but matrix operations might expect (3,1).
    # Based on memory, "PcCircle function requires position and velocity inputs to be at least 2D (e.g., shape (1, 3))".
    # Here we follow MATLAB logic: reshape to 3x1.
    r1 = np.array(r1).reshape(3, 1)
    v1 = np.array(v1).reshape(3, 1)
    r2 = np.array(r2).reshape(3, 1)
    v2 = np.array(v2).reshape(3, 1)

    # Ensure primary/secondary covariances are 6x6
    if C1.shape != (6, 6):
        raise ValueError('Pc3D_Hall:badCovariance: C1 covariance must be a 6x6 matrix')
    if C2.shape != (6, 6):
        raise ValueError('Pc3D_Hall:badCovariance: C2 covariance must be a 6x6 matrix')

    # Process the input HBR
    HBR = np.atleast_1d(HBR)
    N_HBR = HBR.size

    if N_HBR == 2:
        if np.min(HBR) < 0:
            raise ValueError('Pc3D_Hall:InvalidHBR: Both HBR values must be nonnegative.')
        HBR = np.sum(HBR)
    elif N_HBR != 1:
        raise ValueError('Pc3D_Hall:InvalidHBR: Input HBR must have one or two elements.')
    else:
        HBR = HBR[0]

    if HBR <= 0:
        raise ValueError('Pc3D_Hall:InvalidHBR: Combined HBR value must be positive.')

    out['HBR'] = HBR

    # Return 1 for Pc if HBR is infinite
    if np.isinf(HBR):
        return 1.0, out

    # Initialize other misc variables
    H = HBR / 1e3
    H2 = H**2 # HBR in km
    Lclip = (params['Fclip'] * H)**2
    twopi = 2 * np.pi
    twopicubed = twopi**3

    MD2cut = 1491
    Ncdottiny = 1e-300

    # Retrograde orbit processing
    if params['RetrogradeReorientation'] > 0:
        if params['debug_instrumentation']:
            dump_data('Inputs: RetrogradeReorientation', {'r1': r1, 'v1': v1, 'C1': C1, 'r2': r2, 'v2': v2, 'C2': C2, 'params': params}, params['debug_output_file'])
        # RetrogradeReorientation returns tuple (r1, v1, C1, r2, v2, C2, RRout)
        r1, v1, C1, r2, v2, C2, RRout = RetrogradeReorientation(r1, v1, C1, r2, v2, C2, params)
        if params['debug_instrumentation']:
            dump_data('Outputs: RetrogradeReorientation', {'r1': r1, 'v1': v1, 'C1': C1, 'r2': r2, 'v2': v2, 'C2': C2, 'RRout': RRout}, params['debug_output_file'])
        out['RetrogradeReorientation'] = RRout['Reoriented']
    else:
        out['RetrogradeReorientation'] = False

    # Mean equinoctial matrices at nominal TCA
    # EquinoctialMatrices returns X, P, E, J, K, Q, QRemStat, QRaw, QRem, CRem
    # Python EquinoctialMatrices returns flattened X (6,), P (6,6), E (6,), J(6,6), K(6,6), Q(6,6), ...
    # We pass flattened r/v to EquinoctialMatrices as required by its docstring, but reshape back if needed?
    # Actually EquinoctialMatrices takes array-like and flattens them.

    if params['debug_instrumentation']:
        dump_data('Inputs: EquinoctialMatrices (Primary)', {'r1': r1, 'v1': v1, 'C1': C1, 'rem_flag': params['remediate_NPD_TCA_eq_covariances']}, params['debug_output_file'])
    (out['Xmean10'], out['Pmean10'], out['Emean10'], out['Jmean10'], out['Kmean10'],
     out['Qmean10'], out['Qmean10RemStat'], out['Qmean10Raw'],
     out['Qmean10Rem'], C1Rem) = EquinoctialMatrices(r1.flatten(), v1.flatten(), C1,
                                                    params['remediate_NPD_TCA_eq_covariances'])
    if params['debug_instrumentation']:
        dump_data('Outputs: EquinoctialMatrices (Primary)', {'Xmean10': out['Xmean10'], 'Pmean10': out['Pmean10'], 'Emean10': out['Emean10'], 'Jmean10': out['Jmean10'], 'Kmean10': out['Kmean10'], 'Qmean10': out['Qmean10'], 'C1Rem': C1Rem}, params['debug_output_file'])

    if params['debug_instrumentation']:
        dump_data('Inputs: EquinoctialMatrices (Secondary)', {'r2': r2, 'v2': v2, 'C2': C2, 'rem_flag': params['remediate_NPD_TCA_eq_covariances']}, params['debug_output_file'])
    (out['Xmean20'], out['Pmean20'], out['Emean20'], out['Jmean20'], out['Kmean20'],
     out['Qmean20'], out['Qmean20RemStat'], out['Qmean20Raw'],
     out['Qmean20Rem'], C2Rem) = EquinoctialMatrices(r2.flatten(), v2.flatten(), C2,
                                                    params['remediate_NPD_TCA_eq_covariances'])
    if params['debug_instrumentation']:
        dump_data('Outputs: EquinoctialMatrices (Secondary)', {'Xmean20': out['Xmean20'], 'Pmean20': out['Pmean20'], 'Emean20': out['Emean20'], 'Jmean20': out['Jmean20'], 'Kmean20': out['Kmean20'], 'Qmean20': out['Qmean20'], 'C2Rem': C2Rem}, params['debug_output_file'])

    # Return unconverged if any equinoctial elements are undefined (NaN)
    if np.any(np.isnan(out['Emean10'])) or np.any(np.isnan(out['Emean20'])):
        out['converged'] = False
        return np.nan, out

    # Get covariance cross correlation parameters
    if params['apply_covXcorr_corrections']:
        # get_covXcorr_parameters returns (XCprocessing, sigp, Gp, sigs, Gs)
        XCprocessing, sigp, Gp, sigs, Gs = get_covXcorr_parameters(params)
        if XCprocessing:
            sigpXsigs = sigp * sigs
            # DCP 6x1 sensitivity vectors. In Python Gp, Gs are usually (1,6).
            # MATLAB: Gp is returned as 1x6 row vector from get_covXcorr_parameters.
            # Here: Gp = Gp' / 1e3. So Gp becomes 6x1.
            # get_covXcorr_parameters.py returns Gvecp as (1,6) array.

            Gp = Gp.reshape(6, 1) / 1e3
            Gs = Gs.reshape(6, 1) / 1e3

            if out['RetrogradeReorientation']:
                # RRout.M6 is 6x6
                M6 = RRout['M6']
                Gp = M6 @ Gp
                Gs = M6 @ Gs

            # DCP 6x1 sensitivity vectors for TCA equinoctial states
            # Kmean10 is 6x6
            GEp = out['Kmean10'] @ Gp
            GEs = out['Kmean20'] @ Gs
    else:
        XCprocessing = False

    out['covXcorr_corrections_applied'] = XCprocessing

    # Construct relative state and covariance at nominal TCA
    r = r2 - r1
    v = v2 - v1

    if XCprocessing:
        C = C1Rem + C2Rem - sigpXsigs * (Gs @ Gp.T + Gp @ Gs.T)
    else:
        C = C1Rem + C2Rem

    # Calculate linear-trajectory conjunction bounds
    # conj_bounds_Coppola returns (tau0, tau1, tau0_gam1, tau1_gam1)
    # Inputs: gamma, HBR, r, v, C, verbose
    # r, v should be 3x1 or 1x3.
    if params['debug_instrumentation']:
        dump_data('Inputs: conj_bounds_Coppola', {'gamma': params['gamma'], 'HBR': HBR, 'r': r, 'v': v, 'C': C}, params['debug_output_file'])
    out['tau0'], out['tau1'], out['tau0_gam1'], out['tau1_gam1'] = \
        conj_bounds_Coppola(params['gamma'], HBR, r, v, C, params['verbose'])
    if params['debug_instrumentation']:
        dump_data('Outputs: conj_bounds_Coppola', {'tau0': out['tau0'], 'tau1': out['tau1'], 'tau0_gam1': out['tau0_gam1'], 'tau1_gam1': out['tau1_gam1']}, params['debug_output_file'])

    if np.isinf(out['tau0']) or np.isinf(out['tau1']):
        warnings.warn('Pc3D_Hall:InvalidTimeBounds: Coppola conjunction time bound(s) have infinite value(s)')
    elif np.isnan(out['tau0']) or np.isnan(out['tau1']):
        raise ValueError('Pc3D_Hall:InvalidTimeBounds: Coppola conjunction time bound(s) have NaN value(s).')
    elif np.iscomplex(out['tau0']) or np.iscomplex(out['tau1']):
        # In Python floats are real, complex would be complex128 type.
        # But checks like imag != 0 work if variable is complex type.
        if np.iscomplexobj(out['tau0']) and out['tau0'].imag != 0:
             raise ValueError('Pc3D_Hall:InvalidTimeBounds: Coppola conjunction time bound(s) have imaginary component(s).')
    elif out['tau0'] >= out['tau1']:
        raise ValueError('Pc3D_Hall:InvalidTimeBounds: Coppola conjunction time bounds span a nonpositive interval.')

    out['taum'] = (out['tau1'] + out['tau0']) / 2.0
    out['dtau'] = (out['tau1'] - out['tau0'])

    # Orbital periods
    out['period1'] = orbit_period(r1, v1, params['GM'])
    out['period2'] = orbit_period(r2, v2, params['GM'])
    period0 = min(out['period1'], out['period2'])
    half_period0 = 0.5 * period0

    # Initialize parameters for PeakOverlapPos
    PAR = {}
    PAR['verbose'] = params['verbose'] > 1
    PAR['Fclip'] = params['Fclip']
    PAR['maxiter'] = params['POPmaxiter']

    # Create expanded ephemeris conjunction duration time bounds
    # params.Texpand can be scalar or array
    if params['Texpand'] is None:
        Texpand = np.array([])
    else:
        Texpand = np.atleast_1d(params['Texpand'])
    NTexpand = Texpand.size

    # Ncdot duration reduction factor used in convergence diagnostics
    Ncdotgam = max(1e-6, params['gamma'])
    x_val = np.sqrt(2) * scipy.special.erfcinv(Ncdotgam)
    Ncdotred = np.exp(-x_val**2 / 2) / np.sqrt(2 * np.pi)

    bad_Texpand = False

    if NTexpand == 0: # 0 elements -> case 0
        bad_Texpand = False

        Tmin_limit = params['Tmin_limit']
        Tmax_limit = params['Tmax_limit']

        if Tmin_limit is None or Tmax_limit is None:
            Ns = 51
            Ts = min(1.10 * max(out['period1'], out['period2']),
                     2.20 * min(out['period1'], out['period2']))
            Ts = np.linspace(-Ts, Ts, Ns)

            dr2 = delta_r2_equin(Ts,
                out['Emean10'][0], out['Emean10'][1], out['Emean10'][2],
                out['Emean10'][3], out['Emean10'][4], out['Emean10'][5],
                out['Emean20'][0], out['Emean20'][1], out['Emean20'][2],
                out['Emean20'][3], out['Emean20'][4], out['Emean20'][5],
                params['GM'])

            # extrema(x, include_endpoints=True, sort_output=False)
            # MATLAB: extrema(dr2, false, true) -> include_endpoints=false, sort_output=true (descending max, ascending min)
            # extrema returns xmax, imax, xmin, imin.
            # We want indices of maxima.
            # Python extrema: include_endpoints=False, sort_output=True
            _, imax, _, _ = extrema(dr2, include_endpoints=False, sort_output=True)

            # Pick maxima bracketing TCA or Coppola midpoint
            # Ts[imax] are times of maxima.
            # Ia = max(imax(Ts(imax) < min(0,out.taum)));
            # Ib = min(imax(Ts(imax) > max(0,out.taum)));

            # Filter indices
            # Note: extrema returns 0-based indices in Python

            times_max = Ts[imax]

            mask_a = times_max < min(0, out['taum'])
            candidates_a = imax[mask_a]
            Ia = np.max(candidates_a) if candidates_a.size > 0 else None

            mask_b = times_max > max(0, out['taum'])
            candidates_b = imax[mask_b]
            Ib = np.min(candidates_b) if candidates_b.size > 0 else None

            if Ia is None or Ib is None or Ia == 0 or Ib == Ns-1:
                Tmin_limit = -half_period0
                Tmax_limit =  half_period0
            else:
                # Refine bracketing two-body maxima
                # Ia, Ib are indices in Ts.
                # In MATLAB: Ia = Ia-1, Ib = Ib+1 (widening search range, and 1-based indexing adjustment?)
                # Wait, MATLAB: Ia is index in Ts.
                # Ia = max(imax(Ts(imax) < ...)).
                # Then Ia = Ia-1. If Ia was 1, it becomes 0.

                # In Python: Ia is index.
                # Widen window:
                Ia_idx = Ia - 1
                Ib_idx = Ib + 1

                if Ia_idx < 0 or Ib_idx >= Ns:
                     raise ValueError('Failed to find the |r2-r1| maxima bracketing TCA')

                tolX = period0 / 100.0

                def dr2fun(ttt):
                    return delta_r2_equin(ttt,
                        out['Emean10'][0], out['Emean10'][1], out['Emean10'][2],
                        out['Emean10'][3], out['Emean10'][4], out['Emean10'][5],
                        out['Emean20'][0], out['Emean20'][1], out['Emean20'][2],
                        out['Emean20'][3], out['Emean20'][4], out['Emean20'][5],
                        params['GM'])

                # refine_bounded_extrema(fun, x, y, ..., find_minima=False, find_maxima=True, sort_output=True)
                # We need xmnma, ymnma, xmxma, ymxma

                x_slice = Ts[Ia_idx : Ib_idx + 1]
                y_slice = dr2[Ia_idx : Ib_idx + 1]

                # refine_bounded_extrema returns (xmnma, ymnma, xmxma, ymxma, converged, nbisect, x, y, imnma, imxma)

                # Use kwargs for clarity and safety
                refine_verbose = refinement_debug_plotting > 0

                _, _, xmxma, _, _, _, _, _, _, _ = refine_bounded_extrema(
                    fun=dr2fun, x1=x_slice, x2=y_slice, Ninitial=None, Nbisectmax=100, extrema_types=2,
                    TolX=tolX, TolY=np.nan, endpoints=False, verbose=refine_verbose, check_inputs=False
                )

                if xmxma.size < 2:
                    Tmin_limit = -half_period0
                    Tmax_limit =  half_period0
                else:
                    Tmin_limit = np.min(xmxma)
                    Tmax_limit = np.max(xmxma)

                if params['Tmin_limit'] is not None:
                    Tmin_limit = params['Tmin_limit']
                if params['Tmax_limit'] is not None:
                    Tmax_limit = params['Tmax_limit']

                if Tmin_limit >= Tmax_limit:
                    raise ValueError('Pc3D_Hall:InvalidTimeLimits: Calculated min/max conjunction segment time limits invalid')

        if params['Tmin_limit'] is not None:
            Tmin_limit = params['Tmin_limit']
        if params['Tmax_limit'] is not None:
            Tmax_limit = params['Tmax_limit']

        out['Tmin_limit'] = Tmin_limit
        out['Tmax_limit'] = Tmax_limit

        # Initial duration bounds
        if params['Tmin_initial'] is None:
             Tmin_initial = out['tau0']
        elif np.isneginf(params['Tmin_initial']):
             Tmin_initial = Tmin_limit
        else:
             Tmin_initial = params['Tmin_initial']

        if params['Tmax_initial'] is None:
             Tmax_initial = out['tau1']
        elif np.isposinf(params['Tmax_initial']):
             Tmax_initial = Tmax_limit
        else:
             Tmax_initial = params['Tmax_initial']

        out['Tmin_initial'] = Tmin_initial
        out['Tmax_initial'] = Tmax_initial

        out['Neph'] = params['Neph']
        out['Tmin'] = Tmin_initial
        out['Tmax'] = Tmax_initial

        if out['Tmax'] <= Tmin_limit or out['Tmin'] >= Tmax_limit:
            out['Tmin'] = Tmin_limit
            out['Tmax'] = Tmax_limit
        else:
            out['Tmin'] = max(out['Tmin'], Tmin_limit)
            out['Tmax'] = min(out['Tmax'], Tmax_limit)

        starting_at_Coppola_limits = (out['Tmin'] == out['tau0']) and (out['Tmax'] == out['tau1'])

        Nrefinemax = 250

        dTeph0 = (out['Tmax'] - out['Tmin']) / 2.0
        Nrefine0 = Nrefinemax / 3.0
        dTeph0 = max(dTeph0, abs(Tmin_limit)/Nrefine0, abs(Tmax_limit)/Nrefine0)

        Ncdottol = 2e-1
        Ncsumtol = 2e-2
        Ncdotgam = 1e-6
        NcdotgamNeph = 51

        NcdotgamNeph = max(NcdotgamNeph, round((params['Neph'] - 1) / 2))
        Ncdotgam = max(Ncdotgam, params['gamma'])
        x_val = np.sqrt(2) * scipy.special.erfcinv(Ncdotgam)
        Ncdotred = np.exp(-x_val**2 / 2) / np.sqrt(2 * np.pi)

    elif NTexpand == 1: # case 1
        val = Texpand[0]
        if val <= 0 or np.isinf(val) or np.isnan(val):
            bad_Texpand = True
        else:
            bad_Texpand = False
            taue = out['dtau'] / 2.0
            taue = val * taue
            out['Tmin'] = out['taum'] - taue
            out['Tmax'] = out['taum'] + taue

            Tmin_limit = -half_period0
            Tmax_limit =  half_period0
            if params['Torblimit']:
                out['Tmin'] = max(out['Tmin'], Tmin_limit)
                out['Tmax'] = min(out['Tmax'], Tmax_limit)

            if val <= 1:
                out['Neph'] = params['Neph']
            else:
                boost = max(1, taue / half_period0) * val
                Neph_expand = np.ceil(boost * (params['Neph'] - 1) + 1)
                out['Neph'] = int(max(params['Neph'], Neph_expand))
                if out['Neph'] % 2 == 0:
                    out['Neph'] += 1
    elif NTexpand == 2: # case 2
        if np.any(np.isinf(Texpand)) or np.any(np.isnan(Texpand)):
            bad_Texpand = True
        else:
            out['Tmin'] = min(Texpand)
            out['Tmax'] = max(Texpand)

            if out['Tmin'] >= out['Tmax']:
                bad_Texpand = True
            else:
                bad_Texpand = False

            Tmin_limit = -half_period0
            Tmax_limit =  half_period0

            if params['Neph'] < 0:
                out['Neph'] = -params['Neph']
            else:
                Neph_expand = np.ceil(((out['Tmax'] - out['Tmin']) / out['dtau']) * (params['Neph'] - 1) + 1)
                out['Neph'] = int(max(params['Neph'], Neph_expand))
                if out['Neph'] % 2 == 0:
                    out['Neph'] += 1
    else:
        bad_Texpand = True

    if bad_Texpand:
         raise ValueError('Pc3D_Hall:InvalidTExpand: Invalid Texpand parameter')

    out['Texpand'] = Texpand

    # Generate ephemeris times
    out['Teph'] = np.linspace(out['Tmin'], out['Tmax'], out['Neph'])

    # Calc pos/vel mean states and Jacobians
    # jacobian_E0_to_Xt returns (JT, XT) where JT is (NT, 6, 6) and XT is (6, NT)
    if params['debug_instrumentation']:
        dump_data('Inputs: jacobian_E0_to_Xt (Primary)', {'Teph': out['Teph'], 'Emean10': out['Emean10']}, params['debug_output_file'])
    out['Jmean1T'], out['Xmean1T'] = jacobian_E0_to_Xt(out['Teph'], out['Emean10'])
    if params['debug_instrumentation']:
        dump_data('Outputs: jacobian_E0_to_Xt (Primary)', {'Jmean1T': out['Jmean1T'], 'Xmean1T': out['Xmean1T']}, params['debug_output_file'])

    if params['debug_instrumentation']:
        dump_data('Inputs: jacobian_E0_to_Xt (Secondary)', {'Teph': out['Teph'], 'Emean20': out['Emean20']}, params['debug_output_file'])
    out['Jmean2T'], out['Xmean2T'] = jacobian_E0_to_Xt(out['Teph'], out['Emean20'])
    if params['debug_instrumentation']:
        dump_data('Outputs: jacobian_E0_to_Xt (Secondary)', {'Jmean2T': out['Jmean2T'], 'Xmean2T': out['Xmean2T']}, params['debug_output_file'])

    # Initialize output arrays
    # In Python we can just use lists and append or preallocate
    # Using preallocation for performance

    out['Ncdot'] = np.full(out['Neph'], np.nan)
    out['LebNumb'] = np.full(out['Neph'], np.nan)
    out['Ncdot_SmallHBR'] = np.full(out['Neph'], np.nan)
    out['MD2eff'] = np.full(out['Neph'], np.nan)
    out['MS2eff'] = np.full(out['Neph'], np.nan)
    out['POPconv'] = np.full(out['Neph'], True, dtype=bool)
    out['POPiter'] = np.zeros(out['Neph'], dtype=int)
    out['POPfail'] = np.zeros(out['Neph'], dtype=int)

    out['xs1'] = np.full((6, out['Neph']), np.nan)
    out['Es01'] = np.full((6, out['Neph']), np.nan)
    out['Js1'] = np.full((out['Neph'], 6, 6), np.nan) # (NT, 6, 6) convention

    out['xs2'] = np.full((6, out['Neph']), np.nan)
    out['Es02'] = np.full((6, out['Neph']), np.nan)
    out['Js2'] = np.full((out['Neph'], 6, 6), np.nan)

    out['xu'] = np.full((6, out['Neph']), np.nan)
    out['Ps'] = np.full((out['Neph'], 6, 6), np.nan)

    refine_ephemeris = (NTexpand == 0)
    nrefine = 0
    still_refining = True
    need_eph_calc = np.full(out['Neph'], True, dtype=bool)

    while still_refining:
        neph_loop_start = out['Neph']

        if params['verbose']:
            print(f"Neph = {out['Neph']} nrefine = {nrefine} need = {np.sum(need_eph_calc)}")

        for neph in range(out['Neph']):

            if need_eph_calc[neph]:
                need_eph_calc[neph] = False

                PAR['verbose'] = False
                # PeakOverlapPos(t, X1, J1, 0, E1, Q1, X2, J2, 0, E2, Q2, HBR, PAR)
                # Jmean is (NT, 6, 6) in python, so we access [neph, :, :]
                # Jmean in MATLAB was (6, 6, NT).
                if neph == 0 and params['debug_instrumentation']:
                    dump_data('Inputs: PeakOverlapPos (neph=1)', {
                        'Teph': out['Teph'][neph],
                        'Xmean1T': out['Xmean1T'][:, neph],
                        'Jmean1T': out['Jmean1T'][neph, :, :],
                        'Emean10': out['Emean10'],
                        'Qmean10': out['Qmean10'],
                        'Xmean2T': out['Xmean2T'][:, neph],
                        'Jmean2T': out['Jmean2T'][neph, :, :],
                        'Emean20': out['Emean20'],
                        'Qmean20': out['Qmean20'],
                        'H': H,
                        'PAR': PAR
                    }, params['debug_output_file'])

                converged, _, _, _, POP = PeakOverlapPos(out['Teph'][neph],
                    out['Xmean1T'][:, neph], out['Jmean1T'][neph, :, :], 0, out['Emean10'], out['Qmean10'],
                    out['Xmean2T'][:, neph], out['Jmean2T'][neph, :, :], 0, out['Emean20'], out['Qmean20'],
                    H, PAR)

                if neph == 0 and params['debug_instrumentation']:
                    dump_data('Outputs: PeakOverlapPos (neph=1)', {'converged': converged, 'POP': POP}, params['debug_output_file'])

                out['POPconv'][neph] = converged
                out['POPiter'][neph] = POP['iteration']
                out['POPfail'][neph] = POP['failure']

                if converged:
                    out['xs1'][:, neph] = POP['xs1']
                    out['Es01'][:, neph] = POP['Es01']
                    out['Js1'][neph, :, :] = POP['Js1']
                    out['xs2'][:, neph] = POP['xs2']
                    out['Es02'][:, neph] = POP['Es02']
                    out['Js2'][neph, :, :] = POP['Js2']

                    Ps1 = cov_make_symmetric(POP['Js1'] @ out['Qmean10'] @ POP['Js1'].T)
                    Ps2 = cov_make_symmetric(POP['Js2'] @ out['Qmean20'] @ POP['Js2'].T)

                    r1u = POP['xu1'][0:3]
                    v1u = POP['xu1'][3:6]
                    r2u = POP['xu2'][0:3]
                    v2u = POP['xu2'][3:6]

                    ru = r2u - r1u
                    vu = v2u - v1u

                    if XCprocessing:
                        GCp = POP['Js1'] @ GEp
                        GCs = POP['Js2'] @ GEs
                        Ps = Ps1 + Ps2 - sigpXsigs * (GCs @ GCp.T + GCp @ Gs.T)
                    else:
                        Ps = Ps1 + Ps2

                    As = Ps[0:3, 0:3]
                    Bs = Ps[3:6, 0:3]
                    Cs = Ps[3:6, 3:6]

                    # CovRemEigValClip returns dict
                    if neph == 0 and params['debug_instrumentation']:
                        dump_data('Inputs: CovRemEigValClip (neph=1)', {'As': As, 'Lclip': Lclip}, params['debug_output_file'])
                    res = CovRemEigValClip(As, Lclip)
                    Asdet = res['Adet']
                    Asinv = res['Ainv']
                    if neph == 0 and params['debug_instrumentation']:
                        dump_data('Outputs: CovRemEigValClip (neph=1)', {'Asdet': Asdet, 'Asinv': Asinv}, params['debug_output_file'])

                    Ns0 = (twopicubed * Asdet)**(-0.5)
                    bs = Bs @ Asinv
                    Csp = Cs - bs @ Bs.T

                    out['xu'][:, neph] = np.concatenate((ru, vu))
                    out['Ps'][neph, :, :] = Ps

                    Asinv_ru = Asinv @ ru
                    Ms = ru.T @ Asinv_ru
                    Ns = Ns0 * np.exp(-0.5 * Ms)

                    out['Ncdot_SmallHBR'][neph] = H2 * np.pi * Ns * np.linalg.norm(vu)
                    out['MD2eff'][neph] = Ms
                    out['MS2eff'][neph] = Ms - 2 * H * np.linalg.norm(Asinv_ru)

                    if out['MS2eff'][neph] > MD2cut:
                        out['Ncdot'][neph] = 0
                    else:
                        if params['use_Lebedev']:
                            Pint = _Ncdot_integrand(params['vec_Lebedev'], ru, vu, Asinv, H, bs, Csp, -np.log(Ns0), False)
                            # Pint can be array
                            out['Ncdot'][neph] = H2 * np.sum(Pint * params['wgt_Lebedev'])

                            MaxPint = np.max(Pint)
                            out['LebNumb'][neph] = np.sum(Pint > 1e-3 * MaxPint)
                        else:
                            def fun(ph, u):
                                return _Ncdot_quad2d_integrand(ph, u, ru, vu, Asinv, H, bs, Csp, -np.log(Ns0), False)

                            # scipy.integrate.dblquad(func, a, b, gfun, hfun)
                            # limits for ph: 0 to 2pi
                            # limits for u: -1 to 1
                            # dblquad args: func(y, x). Here fun(ph, u). So ph is y, u is x?
                            # quad2d(fun, xmin, xmax, ymin, ymax) -> fun(x, y)
                            # MATLAB: quad2d(fun, 0, twopi, -1, 1). x is ph, y is u.
                            # Scipy dblquad: func(y, x). Inner integral over y.
                            # dblquad(func, a, b, gfun, hfun) computes int_a^b int_gfun(x)^hfun(x) func(y,x) dy dx.
                            # We want int_-1^1 int_0^2pi fun(ph, u) dph du.
                            # So inner integral is dph (ph from 0 to 2pi). Outer is du (u from -1 to 1).
                            # So x is u, y is ph.
                            # func(ph, u).

                            Pint, _ = scipy.integrate.dblquad(fun, -1, 1, lambda x: 0, lambda x: twopi,
                                                              epsabs=params['AbsTol'], epsrel=params['RelTol'])
                            out['Ncdot'][neph] = H2 * Pint

                    if out['Ncdot'][neph] <= Ncdottiny:
                        out['Ncdot'][neph] = 0

        if refine_ephemeris:
            if np.max(out['POPconv']) == 0:
                if out['Tmin'] <= Tmin_limit and out['Tmax'] >= Tmax_limit:
                    if out['Neph'] <= 8 * params['Neph']:
                        out['Neph'] *= 2
                        # Reallocate/Resize handled by add_eph_times logic or explicit re-creation
                        # For simplicity, mimic "Generate a new list of ephemeris times"
                        # We need to preserve structure.
                        # MATLAB code re-generates Teph.

                        out['Teph'] = np.linspace(out['Tmin'], out['Tmax'], out['Neph'])
                        need_eph_calc = np.full(out['Neph'], True, dtype=bool)

                        out['Jmean1T'], out['Xmean1T'] = jacobian_E0_to_Xt(out['Teph'], out['Emean10'])
                        out['Jmean2T'], out['Xmean2T'] = jacobian_E0_to_Xt(out['Teph'], out['Emean20'])

                        # Reset arrays
                        out['Ncdot'] = np.full(out['Neph'], np.nan)
                        out['LebNumb'] = np.full(out['Neph'], np.nan)
                        out['Ncdot_SmallHBR'] = np.full(out['Neph'], np.nan)
                        out['MD2eff'] = np.full(out['Neph'], np.nan)
                        out['MS2eff'] = np.full(out['Neph'], np.nan)
                        out['POPconv'] = np.full(out['Neph'], True, dtype=bool)
                        out['POPiter'] = np.zeros(out['Neph'], dtype=int)
                        out['POPfail'] = np.zeros(out['Neph'], dtype=int)
                        out['xs1'] = np.full((6, out['Neph']), np.nan)
                        out['Es01'] = np.full((6, out['Neph']), np.nan)
                        out['Js1'] = np.full((out['Neph'], 6, 6), np.nan)
                        out['xs2'] = np.full((6, out['Neph']), np.nan)
                        out['Es02'] = np.full((6, out['Neph']), np.nan)
                        out['Js2'] = np.full((out['Neph'], 6, 6), np.nan)
                        out['xu'] = np.full((6, out['Neph']), np.nan)
                        out['Ps'] = np.full((out['Neph'], 6, 6), np.nan)

                    else:
                        still_refining = False
                else:
                    if params['Tmin_initial'] is None or params['Tmax_initial'] is None:
                        if nrefine == 0:
                            stevi = max(abs(out['tau0']), abs(out['tau1']), out['dtau'])
                            out['Tmin'] = max(-stevi, Tmin_limit)
                            out['Tmax'] = min(stevi, Tmax_limit)
                        else:
                            out['Tmin'] = Tmin_limit
                            out['Tmax'] = Tmax_limit
                    else:
                        out['Tmin'] = Tmin_limit
                        out['Tmax'] = Tmax_limit

                    if still_refining:
                        out['Teph'] = np.linspace(out['Tmin'], out['Tmax'], out['Neph'])
                        need_eph_calc = np.full(out['Neph'], True, dtype=bool)
                        out['Jmean1T'], out['Xmean1T'] = jacobian_E0_to_Xt(out['Teph'], out['Emean10'])
                        out['Jmean2T'], out['Xmean2T'] = jacobian_E0_to_Xt(out['Teph'], out['Emean20'])

                        out['Ncdot'] = np.full(out['Neph'], np.nan)
                        out['LebNumb'] = np.full(out['Neph'], np.nan)
                        out['Ncdot_SmallHBR'] = np.full(out['Neph'], np.nan)
                        out['MD2eff'] = np.full(out['Neph'], np.nan)
                        out['MS2eff'] = np.full(out['Neph'], np.nan)
                        out['POPconv'] = np.full(out['Neph'], True, dtype=bool)
                        out['POPiter'] = np.zeros(out['Neph'], dtype=int)
                        out['POPfail'] = np.zeros(out['Neph'], dtype=int)
                        out['xs1'] = np.full((6, out['Neph']), np.nan)
                        out['Es01'] = np.full((6, out['Neph']), np.nan)
                        out['Js1'] = np.full((out['Neph'], 6, 6), np.nan)
                        out['xs2'] = np.full((6, out['Neph']), np.nan)
                        out['Es02'] = np.full((6, out['Neph']), np.nan)
                        out['Js2'] = np.full((out['Neph'], 6, 6), np.nan)
                        out['xu'] = np.full((6, out['Neph']), np.nan)
                        out['Ps'] = np.full((out['Neph'], 6, 6), np.nan)

            else:
                Ncdotmax = np.nanmax(out['Ncdot'])
                nmax = np.nanargmax(out['Ncdot'])
                Ncdotcut = Ncdotmax * Ncdotred

                if Ncdotmax == 0:
                    MD2min = np.nanmin(out['MD2eff'])
                    nmax = np.nanargmin(out['MD2eff'])

                # Refinement logic
                Tnew = []

                if params['AccelerateRefinement'] and nrefine == 0 and starting_at_Coppola_limits \
                   and Ncdotmax > 0 and out['Ncdot'][0] < Ncdotcut and out['Ncdot'][-1] < Ncdotcut \
                   and np.all(out['POPconv']):
                    still_refining = False
                elif nmax == 0 and out['Teph'][0] > Tmin_limit:
                    Tnew.append(max(out['Teph'][0] - dTeph0, Tmin_limit))
                elif nmax == out['Neph'] - 1 and out['Teph'][-1] < Tmax_limit:
                    Tnew.append(min(out['Teph'][-1] + dTeph0, Tmax_limit))
                elif out['Ncdot'][0] > Ncdotcut and out['Teph'][0] > Tmin_limit:
                    Tnew.append(max(out['Teph'][0] - dTeph0, Tmin_limit))
                elif out['Ncdot'][-1] > Ncdotcut and out['Teph'][-1] < Tmax_limit:
                    Tnew.append(min(out['Teph'][-1] + dTeph0, Tmax_limit))
                else:
                    nmd = max(1, min(out['Neph'] - 2, nmax)) # Python index 1 to Neph-2
                    nlo = nmd - 1
                    nhi = nmd + 1

                    if Ncdotmax == 0:
                        if out['MD2eff'][nlo] > out['MD2eff'][nhi]:
                             nbs = nlo; nas = nhi
                        else:
                             nbs = nhi; nas = nlo

                        dTbs = abs(out['Teph'][nmd] - out['Teph'][nbs])
                        dTas = abs(out['Teph'][nmd] - out['Teph'][nas])

                        if dTas > 3 * dTbs:
                            Tnew.append((out['Teph'][nmd] + out['Teph'][nas]) / 2.0)
                        else:
                            dMD2 = abs(out['MD2eff'][nbs] - out['MD2eff'][nmd])
                            if ((dMD2 < 1e-2) or (dMD2 < 1e-2 * MD2min)) and (dTbs < 1e-1 * (out['Tmax'] - out['Tmin'])):
                                still_refining = False
                            else:
                                Tnew.append((out['Teph'][nmd] + out['Teph'][nbs]) / 2.0)
                    else:
                        new_method = True
                        if new_method:
                            dNcd = np.diff(out['Ncdot'])
                            dNcm = np.concatenate(([np.inf], dNcd))
                            dNcp = np.concatenate((-dNcd, [np.inf]))

                            ipeak = (out['Ncdot'] > 0) & (dNcm > 0) & (dNcp > 0)
                            ipeak_indices = np.flatnonzero(ipeak)
                            Npeak = ipeak_indices.size

                            if Npeak > 4:
                                # Sort descending
                                ipsrt = np.argsort(out['Ncdot'][ipeak_indices])[::-1]
                                ipeak_indices = ipeak_indices[ipsrt][:4]
                                Npeak = 4

                            for npeak in range(Npeak):
                                ipk = ipeak_indices[npeak]
                                ipkm1 = ipk - 1
                                ipkp1 = ipk + 1
                                dNctol = Ncdottol * out['Ncdot'][ipk]

                                if ipk > 0:
                                    ibs = ipkm1
                                    dNcbs = out['Ncdot'][ipk] - out['Ncdot'][ibs]
                                    if dNcbs > dNctol:
                                        Tnew.append((out['Teph'][ipk] + out['Teph'][ibs]) / 2.0)
                                    elif ipk < out['Neph'] - 1 and (out['Teph'][ipk] - out['Teph'][ipkm1]) >= 2 * (out['Teph'][ipkp1] - out['Teph'][ipk]):
                                        Tnew.append((out['Teph'][ipk] + out['Teph'][ibs]) / 2.0)

                                if ipk < out['Neph'] - 1:
                                    ibs = ipkp1
                                    dNcbs = out['Ncdot'][ipk] - out['Ncdot'][ibs]
                                    if dNcbs > dNctol:
                                        Tnew.append((out['Teph'][ipk] + out['Teph'][ibs]) / 2.0)
                                    elif ipk > 0 and (out['Teph'][ipkp1] - out['Teph'][ipk]) >= 2 * (out['Teph'][ipk] - out['Teph'][ipkm1]):
                                        Tnew.append((out['Teph'][ipk] + out['Teph'][ibs]) / 2.0)

                        else:
                            if out['Ncdot'][nlo] < out['Ncdot'][nhi]:
                                nbs = nlo
                            else:
                                nbs = nhi
                            if abs(out['Ncdot'][nmd] - out['Ncdot'][nbs]) > Ncdottol * Ncdotmax:
                                Tnew.append((out['Teph'][nmd] + out['Teph'][nbs]) / 2.0)

                    if len(Tnew) > 0:
                         out, need_eph_calc = _add_eph_times(Tnew, out, need_eph_calc)
                    elif still_refining: # Logic check inside MATLAB
                         # Check bisection needed due to large trapezoidal sum
                         df = np.diff(out['Teph'])
                         # dt = [df(1) df(1:end-1)+df(2:end) df(end)]/2
                         dt = np.concatenate(([df[0]], df[:-1] + df[1:], [df[-1]])) / 2.0
                         Ncdt = out['Ncdot'].copy()
                         Ncdt[np.isnan(Ncdt)] = 0
                         st = dt * Ncdt
                         sf = np.sum(st)

                         ndx = st > Ncsumtol * sf
                         if np.any(ndx):
                             ndx_indices = np.flatnonzero(ndx)
                             Tnew = []
                             for n in ndx_indices:
                                 if n == 0:
                                     if out['Teph'][0] > Tmin_limit:
                                         Tnew.append(out['Teph'][n] - dTeph0)
                                     Tnew.append(0.5 * (out['Teph'][n+1] + out['Teph'][n]))
                                 elif n == out['Neph'] - 1:
                                     if out['Teph'][n] < Tmax_limit:
                                         Tnew.append(out['Teph'][n] + dTeph0)
                                     Tnew.append(0.5 * (out['Teph'][n-1] + out['Teph'][n]))
                                 else:
                                     Tnew.append(0.5 * (out['Teph'][n-1] + out['Teph'][n]))
                                     Tnew.append(0.5 * (out['Teph'][n+1] + out['Teph'][n]))

                             out, need_eph_calc = _add_eph_times(Tnew, out, need_eph_calc)
                         else:
                             idx = np.flatnonzero(out['Ncdot'] > Ncdotcut)
                             if idx.size == 0:
                                 Tnew = []
                             else:
                                 # Bisection
                                 idx_list = list(idx)
                                 if idx_list[0] > 0: idx_list.insert(0, idx_list[0]-1)
                                 if idx_list[-1] < out['Neph'] - 1: idx_list.append(idx_list[-1]+1)
                                 idx = np.array(idx_list)

                                 dtmax = (np.max(out['Teph'][idx]) - np.min(out['Teph'][idx])) / NcdotgamNeph
                                 Tnew = []

                                 for n in idx:
                                     if n > 0:
                                         if (out['Teph'][n] - out['Teph'][n-1]) > dtmax:
                                             Tnew.append((out['Teph'][n] + out['Teph'][n-1]) / 2.0)
                                     if n < out['Neph'] - 1:
                                         if (out['Teph'][n+1] - out['Teph'][n]) > dtmax:
                                             Tnew.append((out['Teph'][n+1] + out['Teph'][n]) / 2.0)

                             if len(Tnew) == 0:
                                 still_refining = False
                             else:
                                 out, need_eph_calc = _add_eph_times(Tnew, out, need_eph_calc)

            if still_refining:
                if out['Neph'] == neph_loop_start:
                    still_refining = False
                else:
                    nrefine += 1

            if nrefine > Nrefinemax:
                still_refining = False
        else:
            still_refining = False

    if nrefine > 0:
        out['Tmin'] = np.min(out['Teph'])
        out['Tmax'] = np.max(out['Teph'])

    out['MD2min'] = np.nanmin(out['MD2eff'])
    out['MS2min'] = np.nanmin(out['MS2eff'])

    NcdtSH = out['Ncdot_SmallHBR'].copy()
    NcdtSH[np.isnan(NcdtSH)] = 0
    out['Nccum_SmallHBR'] = scipy.integrate.cumulative_trapezoid(NcdtSH, out['Teph'], initial=0)
    out['Nc_SmallHBR'] = out['Nccum_SmallHBR'][-1]
    if out['Nc_SmallHBR'] <= params['Pc_tiny']:
        out['Nc_SmallHBR'] = 0.0

    Ncdt = out['Ncdot'].copy()
    Ncdt[np.isnan(Ncdt)] = 0
    out['Nccum'] = scipy.integrate.cumulative_trapezoid(Ncdt, out['Teph'], initial=0)
    out['Nc'] = out['Nccum'][-1]
    if out['Nc'] <= params['Pc_tiny']:
        out['Nc'] = 0.0

    POPconv = out['POPconv']

    if np.all(POPconv):
        Ncdotmax = np.nanmax(out['Ncdot'])
        if np.isinf(Ncdotmax):
            out['converged'] = False
            out['Ncdotcut'] = np.nan
        else:
            out['converged'] = True
            out['Ncdotcut'] = Ncdotmax * Ncdotred
    elif not np.any(POPconv):
        out['converged'] = False
        out['Ncdotcut'] = np.nan
    else:
        Ncdotmax = np.nanmax(out['Ncdot'])
        if np.isinf(Ncdotmax):
            out['converged'] = False
            out['Ncdotcut'] = np.nan
        elif np.any(out['POPfail'] >= 100):
            out['converged'] = False
            out['Ncdotcut'] = np.nan
        else:
            out['Ncdotcut'] = Ncdotmax * Ncdotred
            if Ncdotmax == 0:
                if out['MS2min'] > MD2cut:
                    out['converged'] = True
                elif (params['Tmin_limit'] is not None) or (params['Tmax_limit'] is not None):
                     out['converged'] = True
                else:
                    out['converged'] = False
            else:
                idx = np.flatnonzero(out['Ncdot'] >= out['Ncdotcut'])
                out['converged'] = True
                for iii in idx:
                    if iii > 0 and not POPconv[iii-1]:
                        out['converged'] = False
                        break
                    if iii < out['Neph'] - 1 and not POPconv[iii+1]:
                        out['converged'] = False
                        break

    if out['converged'] and out['Nc'] > 0 and params['use_Lebedev']:
        LebNumb = out['LebNumb'].copy()
        LebNumb[np.isnan(LebNumb)] = 0
        AvgLebNumb = scipy.integrate.cumulative_trapezoid(Ncdt * LebNumb, out['Teph'], initial=0)
        out['AvgLebNumb'] = AvgLebNumb[-1] / out['Nc']

    if refine_ephemeris:
        out['converged'] = out['converged'] and (nrefine < Nrefinemax)

    out['nrefine'] = nrefine

    if out['converged']:
        Pc = out['Nc']
        if np.isnan(Pc):
            raise ValueError('Supposedly converged calculation yields Pc = NaN')
        else:
            Pc = min(1.0, Pc)
    else:
        Pc = np.nan

    out['NcdotDurCut'] = Ncdotred
    out['NccumDurCut'] = Ncdotgam

    if not out['converged']:
        out['Ncmaxcut'] = np.nan
        out['Ncmaxima'] = np.nan
        out['Ncminima'] = np.nan
        out['Ncvalmaxima'] = np.nan
        out['NcvalNearTCA'] = np.nan
        out['TaConj'] = np.nan
        out['TbConj'] = np.nan
        out['TpeakConj'] = np.nan
    else:
        if out['Nc_SmallHBR'] == 0:
            NcdtSH = np.exp(-0.5 * (out['MD2eff'] - out['MD2min']))
            NcdtSH[np.isnan(NcdtSH)] = 0

        # extrema(x, include_endpoints, sort_output)
        Ncmaxima, imax, _, _ = extrema(NcdtSH, include_endpoints=True)
        out['Ncmaxima'] = Ncmaxima.size
        _, _, Ncminima, imin = extrema(NcdtSH, include_endpoints=False)
        out['Ncminima'] = Ncminima.size

        if out['Ncmaxima'] > 0:
            if out['Nc'] == 0:
                out['Ncvalmaxima'] = 0
                out['NcvalNearTCA'] = 0
            else:
                out['Ncvalmaxima'] = np.full(out['Ncmaxima'], np.nan)
                # indices imax are relative to NcdtSH array

                for iii in range(out['Ncmaxima']):
                    jmin = imax[iii]
                    for jjj in range(imax[iii], -1, -1):
                        if jjj == 0:
                            jmin = jjj
                            break
                        elif imin.size > 0 and np.min(np.abs(jjj - imin)) == 0:
                             jmin = jjj
                             break

                    jmax = imax[iii]
                    for jjj in range(imax[iii], out['Neph']):
                        if jjj == out['Neph'] - 1:
                            jmax = jjj
                            break
                        elif imin.size > 0 and np.min(np.abs(jjj - imin)) == 0:
                            jmax = max(0, jjj - 1)
                            break

                    if jmin == jmax:
                         if jmin == 0:
                             dTephA = 0.5 * out['Teph'][1]
                         else:
                             dTephA = 0.5 * (out['Teph'][jmin] - out['Teph'][jmin-1])

                         if jmax == out['Neph'] - 1:
                             dTephB = 0.5 * (out['Teph'][jmax] - out['Teph'][jmax-1])
                         else:
                             dTephB = 0.5 * (out['Teph'][jmax+1] - out['Teph'][jmax])

                         Ncvalcum = out['Teph'][jmin] * (dTephA + dTephB)
                    else:
                         slice_range = slice(jmin, jmax+1)
                         Ncvalcum = scipy.integrate.cumulative_trapezoid(Ncdt[slice_range], out['Teph'][slice_range], initial=0)
                         Ncvalcum = Ncvalcum[-1] # scalar

                    out['Ncvalmaxima'][iii] = Ncvalcum

                jjj = np.argmin(np.abs(out['Teph'][imax]))
                out['NcvalNearTCA'] = out['Ncvalmaxima'][jjj]
        else:
            out['Ncvalmaxima'] = np.nan
            out['NcvalNearTCA'] = np.nan

        if out['Ncmaxima'] == 0:
            out['TpeakConj'] = np.nan
        else:
            imax_peak = np.nanargmax(Ncdt)
            out['TpeakConj'] = out['Teph'][imax_peak]

        Ncdtcut = np.nanmax(Ncdt) * out['NcdotDurCut']
        out['Ncmaxcut'] = np.sum(Ncmaxima >= Ncdtcut)

        idx = np.flatnonzero(Ncdt >= Ncdtcut)
        if idx.size > 0:
            out['TaConj'] = np.min(out['Teph'][idx])
            out['TbConj'] = np.max(out['Teph'][idx])
        else:
            out['TaConj'] = np.nan
            out['TbConj'] = np.nan

        if out['Nc'] > 0:
            Nccm = out['Nccum'] / out['Nc']
            ndx = Nccm > 0.5
            Nccm[ndx] = 1.0 - Nccm[ndx]
            ndx = np.flatnonzero(Nccm > out['NccumDurCut'])
            Nndx = ndx.size
            if Nndx > 0:
                if ndx[0] > 0: ndx[0] = ndx[0] - 1
                if ndx[-1] < out['Neph'] - 1: ndx[-1] = ndx[-1] + 1

                # Careful updating TaConj/TbConj
                out['TaConj'] = min(out['TaConj'], np.min(out['Teph'][ndx])) if not np.isnan(out['TaConj']) else np.min(out['Teph'][ndx])
                out['TbConj'] = max(out['TbConj'], np.max(out['Teph'][ndx])) if not np.isnan(out['TbConj']) else np.max(out['Teph'][ndx])

    out['TaFrac'] = out['TaConj'] / Tmin_limit
    out['TbFrac'] = out['TbConj'] / Tmax_limit

    return Pc, out

def _Ncdot_integrand(rht, mur, muv, Ainv, R, Q1, Q2, logZ, slow_method):
    # rht can be (3, N) or (3, N, M)

    rht = np.array(rht)
    sz = rht.shape

    if slow_method:
        # Not implementing slow_method as params default is False
        raise NotImplementedError("slow_method=True not implemented in Python version yet")
    else:
        # rht shape: (3, N) or (3, N1, N2)
        # We need to broadcast.

        # In MATLAB: sznew = cat(2,3,cat(2,1,sz)); rhat = zeros(sznew);
        # This reshaping logic was to use multiprod which handles dims in a specific way.
        # Python's multiprod implementation: "multiprod(A, B, [0, 1])" sums over axis 1.

        # Let's trust broadcasting instead of multiprod if possible, or use multiprod correctly.
        # However, to be safe and consistent with the port, I'll use multiprod logic or equivalent.
        # The variables mur, muv are (3,). Ainv (3,3), R scalar, Q1 (3,3), Q2 (3,3).

        # rht has shape (3, ...). Let's flatten spatial dimensions for easier handling if it is arbitrary.

        orig_shape = rht.shape # (3, ...)
        n_points = np.prod(orig_shape[1:])
        rht_flat = rht.reshape(3, n_points) # (3, N)

        dr = R * rht_flat - mur.reshape(3, 1) # (3, N)

        zero_sig2 = (np.max(np.abs(Q2)) == 0)

        if zero_sig2:
            nu = -np.sum(muv.reshape(3, 1) * rht_flat, axis=0)
            nu[nu < 0] = 0
        else:
            # sig2 = squeeze(multiprod(multitransp(rhat),multiprod(repmat(Q2,szmat),rhat)));
            # rhat' * Q2 * rhat
            # (N, 3) * (3, 3) * (3, N) -> element wise
            # Q2 @ rht_flat -> (3, N)
            # sum(rht_flat * (Q2 @ rht_flat), axis=0) -> (N,)

            sig2 = np.sum(rht_flat * (Q2 @ rht_flat), axis=0)

            nonpos_sig2 = sig2 <= 0
            any_nonpos_sig2 = np.any(nonpos_sig2)

            sig = np.zeros_like(sig2)
            sig[~nonpos_sig2] = np.sqrt(sig2[~nonpos_sig2])
            if any_nonpos_sig2:
                sig[nonpos_sig2] = np.nan

            # nu0 = rhat' * (muv + Q1 * dr)
            term2 = muv.reshape(3, 1) + Q1 @ dr
            nu0 = np.sum(rht_flat * term2, axis=0)

            sqrt2 = np.sqrt(2)
            sqrtpi = np.sqrt(np.pi)
            sqrt2pi = sqrt2 * sqrtpi

            nus = nu0 / (sig * sqrt2)
            H_val = np.exp(-nus**2) - sqrtpi * (nus * scipy.special.erfc(nus))
            nu = sig * H_val / sqrt2pi

            if any_nonpos_sig2:
                nu_nonpos = -np.sum(muv.reshape(3, 1) * rht_flat[:, nonpos_sig2], axis=0)
                nu_nonpos[nu_nonpos < 0] = 0
                nu[nonpos_sig2] = nu_nonpos

        # MD2 = dr' * Ainv * dr
        MD2 = np.sum(dr * (Ainv @ dr), axis=0)

        neglogN3 = logZ + 0.5 * MD2
        integrand = nu * np.exp(-neglogN3)

        # Reshape back
        if len(orig_shape) > 1:
            integrand = integrand.reshape(orig_shape[1:])

        return integrand

def _Ncdot_quad2d_integrand(ph, u, mur, muv, Ainv, R, Q1, Q2, logZ, slow_method):
    # ph, u are arrays of same shape (usually from dblquad/quad2d)
    # If they are scalars, make them 1d arrays
    ph = np.atleast_1d(ph)
    u = np.atleast_1d(u)

    up = np.real(np.sqrt(1 - u**2 + 0j)) # +0j to force complex if negative, then real part
    # Actually u is in [-1, 1], so 1-u^2 >= 0. But for numerical safety.

    rhat = np.zeros((3,) + ph.shape)
    rhat[0, ...] = np.cos(ph) * up
    rhat[1, ...] = np.sin(ph) * up
    rhat[2, ...] = u

    integrand = _Ncdot_integrand(rhat, mur, muv, Ainv, R, Q1, Q2, logZ, slow_method)

    if integrand.size == 1:
        return integrand.item()
    return integrand

def _add_eph_times(Tnew, out, need_eph_calc):
    Tnew = np.unique(Tnew)
    if Tnew.size == 0:
        return out, need_eph_calc

    # Check for repeated values
    # np.isin(element, test_elements)
    ndx = np.isin(Tnew, out['Teph'])
    if np.any(ndx):
        Tnew = Tnew[~ndx]

    if Tnew.size == 0:
        return out, need_eph_calc

    out['Teph'] = np.concatenate((out['Teph'], Tnew))

    J1new, X1new = jacobian_E0_to_Xt(Tnew, out['Emean10'])
    J2new, X2new = jacobian_E0_to_Xt(Tnew, out['Emean20'])

    out['Xmean1T'] = np.hstack((out['Xmean1T'], X1new))
    # Jmean is (NT, 6, 6)
    out['Jmean1T'] = np.concatenate((out['Jmean1T'], J1new), axis=0)
    out['Xmean2T'] = np.hstack((out['Xmean2T'], X2new))
    out['Jmean2T'] = np.concatenate((out['Jmean2T'], J2new), axis=0)

    new_count = Tnew.size
    need_eph_calc = np.concatenate((need_eph_calc, np.full(new_count, True, dtype=bool)))

    # Resize other arrays
    # 1D arrays
    out['Ncdot'] = np.concatenate((out['Ncdot'], np.full(new_count, np.nan)))
    out['LebNumb'] = np.concatenate((out['LebNumb'], np.full(new_count, np.nan)))
    out['Ncdot_SmallHBR'] = np.concatenate((out['Ncdot_SmallHBR'], np.full(new_count, np.nan)))
    out['MD2eff'] = np.concatenate((out['MD2eff'], np.full(new_count, np.nan)))
    out['MS2eff'] = np.concatenate((out['MS2eff'], np.full(new_count, np.nan)))
    out['POPconv'] = np.concatenate((out['POPconv'], np.full(new_count, False, dtype=bool)))
    out['POPiter'] = np.concatenate((out['POPiter'], np.full(new_count, 0, dtype=int))) # NaN int? use 0 or something
    out['POPfail'] = np.concatenate((out['POPfail'], np.full(new_count, 0, dtype=int)))

    # 2D (6, N)
    out['xs1'] = np.hstack((out['xs1'], np.full((6, new_count), np.nan)))
    out['Es01'] = np.hstack((out['Es01'], np.full((6, new_count), np.nan)))
    out['xs2'] = np.hstack((out['xs2'], np.full((6, new_count), np.nan)))
    out['Es02'] = np.hstack((out['Es02'], np.full((6, new_count), np.nan)))
    out['xu'] = np.hstack((out['xu'], np.full((6, new_count), np.nan)))

    # 3D (N, 6, 6)
    out['Js1'] = np.concatenate((out['Js1'], np.full((new_count, 6, 6), np.nan)), axis=0)
    out['Js2'] = np.concatenate((out['Js2'], np.full((new_count, 6, 6), np.nan)), axis=0)
    out['Ps'] = np.concatenate((out['Ps'], np.full((new_count, 6, 6), np.nan)), axis=0)

    # Sort
    nsrt = np.argsort(out['Teph'])
    out['Teph'] = out['Teph'][nsrt]
    need_eph_calc = need_eph_calc[nsrt]

    out['Xmean1T'] = out['Xmean1T'][:, nsrt]
    out['Jmean1T'] = out['Jmean1T'][nsrt, :, :]
    out['Xmean2T'] = out['Xmean2T'][:, nsrt]
    out['Jmean2T'] = out['Jmean2T'][nsrt, :, :]

    out['Ncdot'] = out['Ncdot'][nsrt]
    out['LebNumb'] = out['LebNumb'][nsrt]
    out['Ncdot_SmallHBR'] = out['Ncdot_SmallHBR'][nsrt]
    out['MD2eff'] = out['MD2eff'][nsrt]
    out['MS2eff'] = out['MS2eff'][nsrt]
    out['POPconv'] = out['POPconv'][nsrt]
    out['POPiter'] = out['POPiter'][nsrt]
    out['POPfail'] = out['POPfail'][nsrt]

    out['xs1'] = out['xs1'][:, nsrt]
    out['Es01'] = out['Es01'][:, nsrt]
    out['Js1'] = out['Js1'][nsrt, :, :]
    out['xs2'] = out['xs2'][:, nsrt]
    out['Es02'] = out['Es02'][:, nsrt]
    out['Js2'] = out['Js2'][nsrt, :, :]
    out['xu'] = out['xu'][:, nsrt]
    out['Ps'] = out['Ps'][nsrt, :, :]

    out['Neph'] = out['Teph'].size

    return out, need_eph_calc

def dump_data(label, data, filename):
    import sys
    try:
        # Set print options to avoid truncation of large arrays
        np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)
        with open(filename, 'a') as f:
            f.write(f"=== {label} ===\n")
            for key, val in data.items():
                f.write(f"--- {key} ---\n")
                # Using str(val) or repr(val) might be sufficient, but let's try to match MATLAB's 'disp' style
                # For numpy arrays, default str() is decent.
                f.write(f"{val}\n")
            f.write(f"=== End {label} ===\n")
        # Reset print options to default (optional but polite)
        np.set_printoptions(threshold=1000, linewidth=75)
    except Exception as e:
        # Fallback to console if file write fails, or just ignore?
        # MATLAB version catches evalc errors but might not catch file open errors in the same way.
        # We'll print to stderr for safety.
        import sys
        sys.stderr.write(f"Error writing debug data: {e}\n")
