import numpy as np
import scipy.special
import warnings

# Utils imports
from DistributedPython.ProbabilityOfCollision.Utils.set_default_param import set_default_param
from DistributedPython.ProbabilityOfCollision.Utils.get_covXcorr_parameters import get_covXcorr_parameters
from DistributedPython.ProbabilityOfCollision.Utils.FindNearbyCA import FindNearbyCA
from DistributedPython.Utils.OrbitTransformations.orbit_period import orbit_period

# Pc imports
from DistributedPython.ProbabilityOfCollision.PcCircleWithConjData import PcCircleWithConjData
from DistributedPython.ProbabilityOfCollision.Utils.UsageViolationPc2D import UsageViolationPc2D
from DistributedPython.ProbabilityOfCollision.Pc2D_Hall import Pc2D_Hall
from DistributedPython.ProbabilityOfCollision.Pc3D_Hall import Pc3D_Hall


def PcMultiStep(r1, v1, C1, r2, v2, C2, HBR, params=None):
    """
    Calculate the collision probability for a conjunction using a
    multi-tiered algorithm.

    Parameters
    ----------
    r1, v1, r2, v2 : array_like
        State vectors (m, m/s). 1D arrays or shapes (3,), (1,3), (3,1).
    C1, C2 : array_like
        Covariance matrices (m, m/s). 6x6.
    HBR : float or array_like
        Hard-body radius (m).
    params : dict, optional
        Parameter dictionary.

    Returns
    -------
    Pc : float
        Recommended Pc value.
    out : dict
        Auxiliary output information.
    """

    # Initializations and defaults
    if params is None:
        params = {}

    # Default parameters
    params = set_default_param(params, 'apply_covXcorr_corrections', True)
    params = set_default_param(params, 'apply_TCAoffset_corrections', True)
    params = set_default_param(params, 'InputPc2DValue', None)
    params = set_default_param(params, 'Nc2DParams', {})
    params = set_default_param(params, 'Nc3DParams', {})
    params = set_default_param(params, 'Pc_tiny', 1e-300)

    # Flags to force Pc calculations
    params = set_default_param(params, 'ForcePc2DCalculation', False)
    params = set_default_param(params, 'ForceNc2DCalculation', False)
    params = set_default_param(params, 'ForceNc3DCalculation', False)

    # Flags to only run 2D-Pc calculation
    params = set_default_param(params, 'OnlyPc2DCalculation', False)

    # Flags to prevent 2D-Nc or 3D-Nc
    params = set_default_param(params, 'PreventNc2DCalculation', False)
    # 3D-Nc must also be prevented if 2D-Nc is also prevented
    params = set_default_param(params, 'PreventNc3DCalculation', params['PreventNc2DCalculation'])

    # Flags to perform full 2D-Pc method usage violation analysis
    params = set_default_param(params, 'FullPc2DViolationAnalysis', True)
    params = set_default_param(params, 'AccelerateNc2DCalculation', True)

    # Cutoff factors for 2D-Pc method usage violations
    params = set_default_param(params, 'Pc2DExtendedCutoff', 0.02)
    params = set_default_param(params, 'Pc2DOffsetCutoff', 0.01)
    params = set_default_param(params, 'Pc2DInaccurateCutoff', 0.02)
    params = set_default_param(params, 'AllowUseOfPc2DProxyEstimate', True)
    params = set_default_param(params, 'Pc2DProxyEstimateCutoff', 1e-10)

    # Cutoff factors for 2D-Nc method usage violations
    params = set_default_param(params, 'Nc2DExtendedCutoff', 0.05)
    params = set_default_param(params, 'Nc2DOffsetCutoff', 0.1)
    params = set_default_param(params, 'Nc2DInaccurateCutoff', 0.1)

    # Cutoff factors for 3D-Nc method usage violations
    params = set_default_param(params, 'Nc3DExtendedCutoff', 0.5)
    params = set_default_param(params, 'Nc3DOffsetCutoff', 0.75)
    params = set_default_param(params, 'Nc3DInaccurateCutoff', [0.01, 1e-15])

    # Retrograde orbit reorientation mode
    params = set_default_param(params, 'RetrogradeReorientation', 1)

    # Invalid velocity covariance checking level
    params = set_default_param(params, 'CovVelCheckLevel', 2)

    # Ensure 2D-Nc is calculated, if 3D-Nc is forced
    if params['ForceNc3DCalculation']:
        params['ForceNc2DCalculation'] = True

    # Check the prevent/force parameters
    if params['PreventNc2DCalculation'] and \
            (params['ForceNc2DCalculation'] or params['ForceNc3DCalculation']):
        raise ValueError('Cannot prevent 2D-Nc when either 2D-Nc or 3D-Nc are forced to run.')

    if params['PreventNc3DCalculation'] and params['ForceNc3DCalculation']:
        raise ValueError('3D-Nc cannot be both prevented and forced to run.')

    # Ensure primary/secondary state vectors are row vectors (1x3) for finding CA?
    # Python utils handle shapes generally. But let's standardise.
    r1 = np.array(r1).reshape(1, 3)
    v1 = np.array(v1).reshape(1, 3)
    r2 = np.array(r2).reshape(1, 3)
    v2 = np.array(v2).reshape(1, 3)
    C1 = np.array(C1)
    C2 = np.array(C2)

    if C1.shape != (6, 6):
        raise ValueError('C1 covariance must be a 6x6 matrix')
    if C2.shape != (6, 6):
        raise ValueError('C2 covariance must be a 6x6 matrix')

    # Process input HBR
    HBR = np.atleast_1d(HBR)
    if HBR.size == 2:
        if np.min(HBR) < 0:
            raise ValueError('Both HBR values must be nonnegative.')
        HBR = np.sum(HBR)
    elif HBR.size != 1:
        raise ValueError('Input HBR must have one or two elements.')
    else:
        HBR = HBR[0]

    if HBR <= 0:
        raise ValueError('Combined HBR value must be positive.')

    # Initialize outputs
    Pc = np.nan
    out = {}
    out['Pc2D'] = np.nan
    out['Nc2D'] = np.nan
    out['Nc3D'] = np.nan
    out['PcMethod'] = 'None'
    out['PcMethodNum'] = np.nan
    out['PcMethodMax'] = np.nan

    # Populate default values
    out['IsRemediated'] = None
    out['IsPosDef'] = None
    out['SemiMajorAxis'] = None
    out['SemiMinorAxis'] = None
    out['ClockAngle'] = None
    out['HBR'] = None
    out['MissDistance'] = None
    out['x1Sigma'] = None
    out['RadialSigma'] = None
    out['InTrackSigma'] = None
    out['CrossTrackSigma'] = None
    out['CondNumPrimary'] = None
    out['CondNumSecondary'] = None
    out['CondNumCombined'] = None
    out['CondNumProjected'] = None
    out['RelativePhaseAngle'] = None

    # Save parameters
    out['MultiStepParams'] = params

    # Get covariance cross correction parameters
    out['covXcorr'] = {}
    if params['apply_covXcorr_corrections']:
        res = get_covXcorr_parameters(params)
        out['covXcorr']['Processing'] = res[0]
        out['covXcorr']['sigp'] = res[1]
        out['covXcorr']['Gp'] = res[2]
        out['covXcorr']['sigs'] = res[3]
        out['covXcorr']['Gs'] = res[4]
    else:
        out['covXcorr']['Processing'] = False

    # Check for data quality errors
    q_error_msg = 'Unable to calculate Pc due to invalid 3x3 covariance matrices:'
    msg_len = len(q_error_msg)
    out['DataQualityError'] = {}

    rEarth = 6378135.0
    defaultCovCutoff = (rEarth * 0.99 * 10)**2

    out['DataQualityError']['defaultCovPri'] = np.any(np.diag(C1[0:3, 0:3]) >= defaultCovCutoff)
    if out['DataQualityError']['defaultCovPri']:
        q_error_msg += ' Primary Cov >= Default Covariance Cutoff'

    out['DataQualityError']['defaultCovSec'] = np.any(np.diag(C2[0:3, 0:3]) >= defaultCovCutoff)
    if out['DataQualityError']['defaultCovSec']:
        if len(q_error_msg) > msg_len:
            q_error_msg += ','
        q_error_msg += ' Secondary Cov >= Default Covariance Cutoff'

    out['DataQualityError']['defaultCov'] = out['DataQualityError']['defaultCovPri'] or \
                                            out['DataQualityError']['defaultCovSec']

    out['DataQualityError']['invalidCovPri'] = np.any(np.diag(C1) < 0)
    if out['DataQualityError']['invalidCovPri']:
        if len(q_error_msg) > msg_len:
            q_error_msg += ','
        q_error_msg += ' Primary Cov has negative sigma^2 value'

    out['DataQualityError']['invalidCovSec'] = np.any(np.diag(C2) < 0)
    if out['DataQualityError']['invalidCovSec']:
        if len(q_error_msg) > msg_len:
            q_error_msg += ','
        q_error_msg += ' Secondary Cov has negative sigma^2 value'

    out['DataQualityError']['invalidCov'] = out['DataQualityError']['invalidCovPri'] or \
                                            out['DataQualityError']['invalidCovSec']

    out['DataQualityError']['invalidCov6x6Pri'] = False
    out['DataQualityError']['invalidCov6x6Sec'] = False
    out['DataQualityError']['invalidCov6x6'] = False

    zeros3x1 = np.zeros(3)
    out['DataQualityError']['invalidCov3x3Pri'] = np.array_equal(zeros3x1, np.diag(C1[0:3, 0:3]))
    if out['DataQualityError']['invalidCov3x3Pri']:
        if len(q_error_msg) > msg_len:
            q_error_msg += ','
        q_error_msg += ' Primary Cov contains position components with values of zero'

    out['DataQualityError']['invalidCov3x3Sec'] = np.array_equal(zeros3x1, np.diag(C2[0:3, 0:3]))
    if out['DataQualityError']['invalidCov3x3Sec']:
        if len(q_error_msg) > msg_len:
            q_error_msg += ','
        q_error_msg += ' Secondary Cov contains position components with values of zero'

    out['DataQualityError']['invalidCov3x3'] = out['DataQualityError']['invalidCov3x3Pri'] or \
                                               out['DataQualityError']['invalidCov3x3Sec']

    out['AnyDataQualityErrorsPri'] = out['DataQualityError']['defaultCovPri'] or \
                                     out['DataQualityError']['invalidCovPri'] or \
                                     out['DataQualityError']['invalidCov3x3Pri']
    out['AnyDataQualityErrorsSec'] = out['DataQualityError']['defaultCovSec'] or \
                                     out['DataQualityError']['invalidCovSec'] or \
                                     out['DataQualityError']['invalidCov3x3Sec']
    out['AnyDataQualityErrors'] = out['DataQualityError']['defaultCov'] or \
                                  out['DataQualityError']['invalidCov'] or \
                                  out['DataQualityError']['invalidCov3x3']

    if out['AnyDataQualityErrors']:
        out['PcMethod'] = out['PcMethod'] + q_error_msg
        return Pc, out

    # Perform 2D-Pc processing
    if params['ForcePc2DCalculation']:
        NeedPc2DCalculation = True
    else:
        if params['InputPc2DValue'] is None:
            NeedPc2DCalculation = True
        else:
            NeedPc2DCalculation = np.isnan(params['InputPc2DValue'])

    out['Pc2DInfo'] = {}
    if not NeedPc2DCalculation:
        out['Pc2D'] = params['InputPc2DValue']
        out['Pc2DInfo']['Method'] = '2D-Pc (value input)'
    else:
        out['PcMethodMax'] = 1.0 # 1.0 = 2D-Pc

        if params['apply_TCAoffset_corrections']:
            X1 = np.hstack((r1, v1)).T # (6, 1)
            X2 = np.hstack((r2, v2)).T # (6, 1)
            # FindNearbyCA expects (6, 1) inputs? Or array-like?
            # It returns dTCA, X1CA, X2CA
            dTCA, X1CA, X2CA = FindNearbyCA(X1, X2)
            out['Pc2DInfo']['dTCA'] = dTCA
            out['Pc2DInfo']['X1CA'] = X1CA
            out['Pc2DInfo']['X2CA'] = X2CA

            # X1CA is (6, 1)
            r1CA = X1CA[0:3].flatten().reshape(1, 3)
            v1CA = X1CA[3:6].flatten().reshape(1, 3)
            r2CA = X2CA[0:3].flatten().reshape(1, 3)
            v2CA = X2CA[3:6].flatten().reshape(1, 3)
        else:
            out['Pc2DInfo']['dTCA'] = np.nan
            out['Pc2DInfo']['X1CA'] = np.full(6, np.nan)
            out['Pc2DInfo']['X2CA'] = np.full(6, np.nan)
            r1CA = r1.flatten().reshape(1, 3)
            v1CA = v1.flatten().reshape(1, 3)
            r2CA = r2.flatten().reshape(1, 3)
            v2CA = v2.flatten().reshape(1, 3)

        Arel = C1[0:3, 0:3] + C2[0:3, 0:3]
        if out['covXcorr']['Processing']:
            sigpXsigs = out['covXcorr']['sigp'] * out['covXcorr']['sigs']
            Gp = out['covXcorr']['Gp'][0:3].reshape(3, 1) # Python: usually 1D array
            Gs = out['covXcorr']['Gs'][0:3].reshape(3, 1)

            Arel = Arel - sigpXsigs * (Gs @ Gp.T + Gp @ Gs.T)
            out['covXcorr_corrections_applied'] = True

            out['Pc2D'], tempOut = PcCircleWithConjData(r1CA, v1CA, Arel, r2CA, v2CA, np.zeros_like(Arel), HBR)
        else:
            out['covXcorr_corrections_applied'] = False
            out['Pc2D'], tempOut = PcCircleWithConjData(r1CA, v1CA, C1[0:3, 0:3], r2CA, v2CA, C2[0:3, 0:3], HBR)

        out['Pc2DInfo']['Method'] = '2D-Pc'
        out['Pc2DInfo']['Arel'] = Arel

        # Populate info
        out['Pc2DInfo']['Remediated'] = tempOut['IsRemediated']
        out['Pc2DInfo']['xmiss'] = tempOut['xm']
        out['Pc2DInfo']['ymiss'] = tempOut['zm']
        out['Pc2DInfo']['xsigma'] = tempOut['sx']
        out['Pc2DInfo']['ysigma'] = tempOut['sz']
        out['Pc2DInfo']['EigV1'] = tempOut['EigV1']
        out['Pc2DInfo']['EigL1'] = tempOut['EigL1']
        out['Pc2DInfo']['EigV2'] = tempOut['EigV2']
        out['Pc2DInfo']['EigL2'] = tempOut['EigL2']

        # Optional fields from PcCircleWithConjData
        for key in ['EigV1Pri', 'EigL1Pri', 'EigV2Pri', 'EigL2Pri',
                    'EigV1Sec', 'EigL1Sec', 'EigV2Sec', 'EigL2Sec']:
            if key in tempOut:
                out['Pc2DInfo'][key] = tempOut[key]

        # Populate ConjData
        out['IsRemediated'] = tempOut['IsRemediated']
        out['IsPosDef'] = tempOut['IsPosDef']
        out['SemiMajorAxis'] = tempOut['SemiMajorAxis']
        out['SemiMinorAxis'] = tempOut['SemiMinorAxis']
        out['ClockAngle'] = tempOut['ClockAngle']
        out['HBR'] = tempOut['HBR']
        out['MissDistance'] = tempOut['MissDistance']
        out['x1Sigma'] = tempOut['x1Sigma']
        out['RadialSigma'] = tempOut['RadialSigma']
        out['InTrackSigma'] = tempOut['InTrackSigma']
        out['CrossTrackSigma'] = tempOut['CrossTrackSigma']
        out['CondNumPrimary'] = tempOut['CondNumPrimary']
        out['CondNumSecondary'] = tempOut['CondNumSecondary']
        out['CondNumCombined'] = tempOut['CondNumCombined']
        out['CondNumProjected'] = tempOut['CondNumProjected']
        out['RelativePhaseAngle'] = tempOut['RelativePhaseAngle']

    if not np.isnan(out['Pc2D']):
        Pc = out['Pc2D']
        out['PcMethod'] = out['Pc2DInfo']['Method']
        out['PcMethodNum'] = 1.0

    if params['OnlyPc2DCalculation']:
        return Pc, out

    # Perform individual covariance checks
    if params['CovVelCheckLevel'] > 0:
        if np.array_equal(zeros3x1, np.diag(C1[3:6, 3:6])):
            out['DataQualityError']['invalidCov6x6Pri'] = True
            out['AnyDataQualityErrorsPri'] = True
        if np.array_equal(zeros3x1, np.diag(C2[3:6, 3:6])):
            out['DataQualityError']['invalidCov6x6Sec'] = True
            out['AnyDataQualityErrorsSec'] = True

    if params['CovVelCheckLevel'] > 1:
        zeros3x3 = np.zeros((3, 3))
        if np.array_equal(zeros3x3, C1[0:3, 3:6]):
            out['DataQualityError']['invalidCov6x6Pri'] = True
            out['AnyDataQualityErrorsPri'] = True
        if np.array_equal(zeros3x3, C2[0:3, 3:6]):
            out['DataQualityError']['invalidCov6x6Sec'] = True
            out['AnyDataQualityErrorsSec'] = True

    out['DataQualityError']['invalidCov6x6'] = out['DataQualityError']['invalidCov6x6Pri'] or \
                                              out['DataQualityError']['invalidCov6x6Sec']
    out['AnyDataQualityErrors'] = out['AnyDataQualityErrorsPri'] or \
                                  out['AnyDataQualityErrorsSec']

    if out['AnyDataQualityErrors']:
        out['PcMethod'] = out['PcMethod'] + ' (cannot calculate more advanced Pc due to invalid covariance)'
        return Pc, out

    # Perform 2D-Pc usage violation processing
    if not params['FullPc2DViolationAnalysis']:
        if params['ForceNc2DCalculation']:
            pass
        else:
            warnings.warn('Full 2D-Pc usage violation analysis is strongly recommended unless forcing 2D-Nc calculations')

        if params['AccelerateNc2DCalculation']:
            warnings.warn('Accelerated 2D-Nc method calculation is only possible if full 2D-Pc usage violation analysis is performed; resetting AccelerateNc2DCalculation flag')
            params['AccelerateNc2DCalculation'] = False

    out['Pc2DViolations'] = {}
    # UsageViolationPc2D returns UVIndicators, UVInfo
    out['Pc2DViolations']['Indicators'], out['Pc2DViolations']['Info'] = \
        UsageViolationPc2D(r1, v1, C1, r2, v2, C2, HBR, params)

    out['Pc2DViolations']['LogFactor'] = out['Pc2DViolations']['Info']['LogPcCorrectionFactor']

    indicators = out['Pc2DViolations']['Indicators']
    # Check for NaNs
    # indicators is a dict
    inds_array = np.array([
        np.any(indicators['NPDIssues']),
        indicators['Extended'],
        indicators['Offset'],
        indicators['Inaccurate']
    ])

    if np.any(np.isnan(inds_array)):
        out['AllIndicatorsConverged'] = False
    else:
        out['AllIndicatorsConverged'] = True

    if out['AllIndicatorsConverged']:
        out['Pc2DViolations']['NPD'] = np.any(indicators['NPDIssues'])
        out['Pc2DViolations']['Extended'] = indicators['Extended'] > params['Pc2DExtendedCutoff']
        out['Pc2DViolations']['Offset'] = indicators['Offset'] > params['Pc2DOffsetCutoff']
        out['Pc2DViolations']['Inaccurate'] = indicators['Inaccurate'] > params['Pc2DInaccurateCutoff']

        out['AnyPc2DViolations'] = out['Pc2DViolations']['NPD'] or \
                                   out['Pc2DViolations']['Extended'] or \
                                   out['Pc2DViolations']['Offset'] or \
                                   out['Pc2DViolations']['Inaccurate']

        out['Pc2DViolations']['PcProxyEstimate'] = np.nan
        out['Pc2DViolations']['PcScaledEstimate'] = np.nan

        if params['FullPc2DViolationAnalysis']:
            out['Pc2DViolations']['Nc2DSmallHBR'] = out['Pc2DViolations']['Info']['Nc2DSmallHBR']
            out['Pc2DViolations']['Nc2DLoLimit'] = out['Pc2DViolations']['Info']['Nc2DLoLimit']
            out['Pc2DViolations']['Nc2DHiLimit'] = out['Pc2DViolations']['Info']['Nc2DHiLimit']

            if not np.isnan(out['Pc2DViolations']['LogFactor']):
                if out['Pc2DViolations']['LogFactor'] == 0:
                    out['Pc2DViolations']['PcScaledEstimate'] = Pc
                else:
                    PcClip = max(np.finfo(float).tiny, out['Pc2D'])
                    out['Pc2DViolations']['PcScaledEstimate'] = min(1.0, np.exp(out['Pc2DViolations']['LogFactor'] + np.log(PcClip)))

            ProxyNum = 0
            if not np.isnan(out['Pc2DViolations']['Nc2DHiLimit']):
                ProxyNum = 1
                out['Pc2DViolations']['PcProxyEstimate'] = out['Pc2DViolations']['Nc2DHiLimit']
            elif not np.isnan(out['Pc2DViolations']['PcScaledEstimate']):
                ProxyNum = 2
                out['Pc2DViolations']['PcProxyEstimate'] = out['Pc2DViolations']['PcScaledEstimate']

            if out['AnyPc2DViolations'] and params['AllowUseOfPc2DProxyEstimate']:
                out['PcMethodMax'] = 1.5

                if np.isnan(out['Pc2DViolations']['PcProxyEstimate']):
                    out['PcMethod'] = out['PcMethod'] + ' (failed 2D-Pc method usage violation analysis)'
                else:
                    Pc = out['Pc2DViolations']['PcProxyEstimate']
                    if ProxyNum == 1:
                        out['PcMethod'] = '2D-Nc-UpperLimit (to account for 2D-Pc method usage violations)'
                        out['PcMethodNum'] = 1.5
                    elif ProxyNum == 2:
                        out['PcMethod'] = '2D-Pc-Scaled (to account for 2D-Pc method usage violations)'
                        out['PcMethodNum'] = 1.1
    else:
        out['AnyPc2DViolations'] = True

    # Perform 2D-Nc processing
    if params['PreventNc2DCalculation']:
        NeedNc2DCalculation = False
    elif params['ForceNc2DCalculation'] or np.isnan(Pc):
        NeedNc2DCalculation = True
    elif not out['AllIndicatorsConverged']:
        NeedNc2DCalculation = True
    elif out['AnyPc2DViolations']:
        if out['Pc2DViolations']['Extended'] or out['Pc2DViolations']['Offset']:
            NeedNc2DCalculation = True
        else:
            if not params['AllowUseOfPc2DProxyEstimate']:
                NeedNc2DCalculation = True
            else:
                isnanPcCurrent = np.isnan(Pc)
                isnanPcProxy = np.isnan(out['Pc2DViolations']['PcProxyEstimate'])
                if isnanPcCurrent and isnanPcProxy:
                    NeedNc2DCalculation = True
                else:
                    PcCut = max(Pc if not isnanPcCurrent else -np.inf,
                                out['Pc2DViolations']['PcProxyEstimate'] if not isnanPcProxy else -np.inf)
                    NeedNc2DCalculation = PcCut >= params['Pc2DProxyEstimateCutoff']
    else:
        NeedNc2DCalculation = False

    out['Nc2DInfo'] = {}
    if NeedNc2DCalculation:
        out['PcMethodMax'] = 2.0

        Nc2DParams = params['Nc2DParams'].copy()

        if params['apply_covXcorr_corrections']:
            res = get_covXcorr_parameters(params)
            Nc2DParams['apply_covXcorr_corrections'] = True
            Nc2DParams['covXcorr'] = {}
            Nc2DParams['covXcorr']['sigp'] = res[1]
            Nc2DParams['covXcorr']['Gvecp'] = res[2]
            Nc2DParams['covXcorr']['sigs'] = res[3]
            Nc2DParams['covXcorr']['Gvecs'] = res[4]
        else:
            Nc2DParams['apply_covXcorr_corrections'] = False

        if params['AccelerateNc2DCalculation']:
            Nc2DParams['UVPc2D'] = out['Pc2DViolations']['Info']
        else:
            Nc2DParams['UVPc2D'] = None

        Nc2DParams['CalcConjTimes'] = True
        Nc2DParams['RelTolConjPlane'] = max(1e-2, 1e-1 * params['Nc2DInaccurateCutoff'])
        Nc2DParams['RelTolIntegral2'] = max(1e-5, 1e-4 * params['Nc2DInaccurateCutoff'])

        Nc2DParams = set_default_param(Nc2DParams, 'RetrogradeReorientation', params['RetrogradeReorientation'])

        out['Nc2DParams'] = Nc2DParams

        out['Nc2D'], out['Nc2DInfo'] = Pc2D_Hall(r1, v1, C1, r2, v2, C2, HBR, Nc2DParams)

        # Initialize indicator/violations structures if not present (Pc2D_Hall output structure usually has them but logic below adds more)
        if 'Indicators' not in out['Nc2DInfo']: out['Nc2DInfo']['Indicators'] = {}
        if 'Violations' not in out['Nc2DInfo']: out['Nc2DInfo']['Violations'] = {}

        if np.isnan(out['Nc2D']):
            out['Nc2DInfo']['Converged'] = False
            out['AnyNc2DViolations'] = True
        else:
            out['Nc2DInfo']['Converged'] = True

            out['Nc2DInfo']['Period1'] = orbit_period(r1, v1)
            out['Nc2DInfo']['Period2'] = orbit_period(r2, v2)
            PeriodMin = min(out['Nc2DInfo']['Period1'], out['Nc2DInfo']['Period2'])

            TMeanRate = out['Nc2DInfo'].get('TMeanRate', np.nan)
            TSigmaRate = out['Nc2DInfo'].get('TSigmaRate', np.nan)

            if np.isnan(TMeanRate) or np.isnan(TSigmaRate):
                TMeanRate = out['Nc2DInfo'].get('TQ0min', np.nan)
                TSigmaRate = out['Nc2DInfo'].get('SigmaQ0min', np.nan)

            if np.isnan(TMeanRate) or np.isnan(TSigmaRate):
                raise ValueError('No usable parameters to estimate peak Ncdot time and sigma')

            gamma = 1e-16
            dtau = TSigmaRate * np.sqrt(2) * scipy.special.erfcinv(gamma)
            Ta = TMeanRate - dtau
            Tb = TMeanRate + dtau

            out['Nc2DInfo']['Ta'] = Ta
            out['Nc2DInfo']['Tb'] = Tb

            out['Nc2DInfo']['Indicators']['Extended'] = (Tb - Ta) / PeriodMin

            Tab = max(abs(Ta), abs(Tb))
            out['Nc2DInfo']['Indicators']['Offset'] = Tab / PeriodMin

            PcBest = out['Nc2D']
            PcAlt = out['Nc2DInfo'].get('PcAlt', np.nan)

            if np.isnan(PcBest) or np.isnan(PcAlt):
                out['Nc2DInfo']['Indicators']['Inaccurate'] = 2
            elif PcBest == PcAlt:
                out['Nc2DInfo']['Indicators']['Inaccurate'] = 0
            else:
                PcMean = (PcBest + PcAlt) / 2.0
                out['Nc2DInfo']['Indicators']['Inaccurate'] = abs(PcBest - PcAlt) / PcMean

            out['Nc2DInfo']['Violations']['Extended'] = out['Nc2DInfo']['Indicators']['Extended'] > params['Nc2DExtendedCutoff']
            out['Nc2DInfo']['Violations']['Offset'] = out['Nc2DInfo']['Indicators']['Offset'] > params['Nc2DOffsetCutoff']
            out['Nc2DInfo']['Violations']['Inaccurate'] = out['Nc2DInfo']['Indicators']['Inaccurate'] > params['Nc2DInaccurateCutoff']

            out['AnyNc2DViolations'] = out['Nc2DInfo']['Violations']['Extended'] or \
                                       out['Nc2DInfo']['Violations']['Offset'] or \
                                       out['Nc2DInfo']['Violations']['Inaccurate']

        if out['Nc2DInfo']['Converged']:
            Pc = out['Nc2D']
            Pcmethod = out['Nc2DInfo'].get('Pcmethod', 2)
            if Pcmethod == 2:
                out['PcMethodNum'] = 2.0
                out['PcMethod'] = '2D-Nc'
            elif Pcmethod == 3:
                out['PcMethodNum'] = 2.5
                out['PcMethod'] = '2D-Nc-Full (full accuracy unit-sphere integration)'
            elif Pcmethod == 1:
                out['PcMethodNum'] = 2.6
                out['PcMethod'] = '2D-Nc-ConjPlane (conj. plane integration approx.)'
            else:
                raise ValueError('Invalid 2D-Nc method')

    # Perform 3D-Nc processing
    if params['PreventNc3DCalculation']:
        NeedNc3DCalculation = False
    elif params['ForceNc3DCalculation'] or np.isnan(Pc):
        NeedNc3DCalculation = True
    elif NeedNc2DCalculation and out.get('AnyNc2DViolations', False):
        NeedNc3DCalculation = True
    else:
        NeedNc3DCalculation = False

    out['NeedNc3DCalculation'] = NeedNc3DCalculation

    out['Nc3DInfo'] = {}
    if NeedNc3DCalculation:
        out['PcMethodMax'] = 3.0

        if out['PcMethodNum'] == 2.6:
            Use2DNcForLargeHBREstimate = 1
            out['PcMethodMax'] = 3.6
        else:
            Use2DNcForLargeHBREstimate = 0

        Nc3DParams = params['Nc3DParams'].copy()

        if params['apply_covXcorr_corrections']:
            res = get_covXcorr_parameters(params)
            Nc3DParams['apply_covXcorr_corrections'] = True
            Nc3DParams['covXcorr'] = {}
            Nc3DParams['covXcorr']['sigp'] = res[1]
            Nc3DParams['covXcorr']['Gvecp'] = res[2]
            Nc3DParams['covXcorr']['sigs'] = res[3]
            Nc3DParams['covXcorr']['Gvecs'] = res[4]
        else:
            Nc3DParams['apply_covXcorr_corrections'] = False

        Nc2DInfo = out.get('Nc2DInfo', {})
        Nc2DConverged = Nc2DInfo.get('Converged', False)
        Nc2DViolations = Nc2DInfo.get('Violations', {})

        if not Nc2DConverged or \
           Nc2DViolations.get('Extended', False) or \
           Nc2DViolations.get('Offset', False):
            Nc3DParams = set_default_param(Nc3DParams, 'Tmin_initial', -np.inf)
            Nc3DParams = set_default_param(Nc3DParams, 'Tmax_initial', np.inf)
        else:
            Nc3DParams = set_default_param(Nc3DParams, 'Tmin_initial', None)
            Nc3DParams = set_default_param(Nc3DParams, 'Tmax_initial', None)

        if Nc3DParams['Tmin_initial'] is None and Nc3DParams['Tmax_initial'] is None:
            Nc3DSegmentString = ''
        elif np.isneginf(Nc3DParams['Tmin_initial']) and np.isposinf(Nc3DParams['Tmax_initial']):
            Nc3DSegmentString = '-Full (full encounter segment integration)'
            out['PcMethodMax'] = 3.5
        else:
            Nc3DSegmentString = ' (over input time bounds)'

        Nc3DParams = set_default_param(Nc3DParams, 'RetrogradeReorientation', params['RetrogradeReorientation'])

        out['Nc3DParams'] = Nc3DParams

        out['Nc3D'], out['Nc3DInfo'] = Pc3D_Hall(r1, v1, C1, r2, v2, C2, HBR, Nc3DParams)

        # Initialize indicator/violations structures
        if 'Indicators' not in out['Nc3DInfo']: out['Nc3DInfo']['Indicators'] = {}
        if 'Violations' not in out['Nc3DInfo']: out['Nc3DInfo']['Violations'] = {}

        if np.isnan(out['Nc3D']) or not out['Nc3DInfo'].get('converged', False):
            out['Nc3DInfo']['Converged'] = False
            out['AnyNc3DViolations'] = True
        else:
            out['Nc3DInfo']['Converged'] = True

            TA = out['Nc3DInfo']['Tmin_limit']
            TB = out['Nc3DInfo']['Tmax_limit']
            Ta = out['Nc3DInfo']['TaConj']
            Tb = out['Nc3DInfo']['TbConj']

            if TB != TA:
                out['Nc3DInfo']['Indicators']['Extended'] = (Tb - Ta) / (TB - TA)
            else:
                out['Nc3DInfo']['Indicators']['Extended'] = 0 # Or NaN?

            out['Nc3DInfo']['Violations']['Extended'] = out['Nc3DInfo']['Indicators']['Extended'] > params['Nc3DExtendedCutoff']

            out['Nc3DInfo']['Indicators']['Offset'] = max(Ta / TA if TA!=0 else 0, Tb / TB if TB!=0 else 0)

            out['Nc3DInfo']['Violations']['Offset'] = out['Nc3DInfo']['Indicators']['Offset'] > params['Nc3DOffsetCutoff']

            if 'AvgLebNumb' not in out['Nc3DInfo'] or np.isnan(out['Nc3DInfo']['AvgLebNumb']):
                out['Nc3DInfo']['Indicators']['Inaccurate'] = np.nan
                out['Nc3DInfo']['Violations']['Inaccurate'] = False
            else:
                MinAvgLebNum = params['Nc3DInaccurateCutoff'][0] * out['Nc3DInfo']['params']['deg_Lebedev']
                out['Nc3DInfo']['Indicators']['Inaccurate'] = max(0, (MinAvgLebNum - out['Nc3DInfo']['AvgLebNumb']) / MinAvgLebNum)

                out['Nc3DInfo']['Violations']['Inaccurate'] = out['Nc3DInfo']['Indicators']['Inaccurate'] > params['Nc3DInaccurateCutoff'][0]

                if out['Nc3DInfo']['Violations']['Inaccurate'] and not Use2DNcForLargeHBREstimate and out['PcMethodNum'] >= 2:
                    Nc2DBest = out['Nc2D']
                    Nc2DAlt = out['Nc2DInfo'].get('PcCP', np.nan)

                    if not np.isnan(Nc2DBest) and not np.isnan(Nc2DAlt):
                        if Nc2DBest == Nc2DAlt:
                            InaccIndicator = 0
                        else:
                            Nc2DMean = (Nc2DBest + Nc2DAlt) / 2.0
                            InaccIndicator = abs(Nc2DBest - Nc2DAlt) / Nc2DMean

                        if InaccIndicator <= params['Nc2DInaccurateCutoff']:
                            Use2DNcForLargeHBREstimate = 2
                            out['Nc3DInfo']['Indicators']['Inaccurate'] = InaccIndicator
                            out['Nc3DInfo']['Violations']['Inaccurate'] = False
                            out['PcMethodMax'] = 3.6
                elif out['Nc3D'] < params['Nc3DInaccurateCutoff'][1]:
                    out['Nc3DInfo']['Violations']['Inaccurate'] = False

            out['AnyNc3DViolations'] = out['Nc3DInfo']['Violations']['Extended'] or \
                                       out['Nc3DInfo']['Violations']['Offset'] or \
                                       out['Nc3DInfo']['Violations']['Inaccurate']

        if Use2DNcForLargeHBREstimate > 0:
            out['Nc3D'] = out['Nc2D']
            Pc = out['Nc2D']
            out['PcMethodNum'] = out['PcMethodMax']
            out['PcMethod'] = '3D-Nc-ConjPlane (conj. plane integration approx.)'
            out['Nc3DInfo']['Use2DNcForLargeHBREstimate'] = Use2DNcForLargeHBREstimate

            if Use2DNcForLargeHBREstimate == 1:
                out['Nc3DInfo']['Indicators']['Inaccurate'] = out['Nc2DInfo']['Indicators']['Inaccurate']
                out['Nc3DInfo']['Violations']['Inaccurate'] = out['Nc2DInfo']['Violations']['Inaccurate']
                out['AnyNc3DViolations'] = out['Nc3DInfo']['Violations']['Extended'] or \
                                           out['Nc3DInfo']['Violations']['Offset'] or \
                                           out['Nc3DInfo']['Violations']['Inaccurate']
        elif out['Nc3DInfo']['Converged']:
            Pc = out['Nc3D']
            out['PcMethod'] = '3D-Nc' + Nc3DSegmentString
            out['PcMethodNum'] = out['PcMethodMax']

    if not np.isnan(Pc) and Pc <= params['Pc_tiny']:
        Pc = 0.0

    return Pc, out
