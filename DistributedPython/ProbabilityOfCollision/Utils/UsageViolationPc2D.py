import numpy as np
from scipy.special import erfcinv
from DistributedPython.ProbabilityOfCollision.Utils.set_default_param import set_default_param
from DistributedPython.ProbabilityOfCollision.Utils.RetrogradeReorientation import RetrogradeReorientation
from DistributedPython.ProbabilityOfCollision.Utils.EquinoctialMatrices import EquinoctialMatrices
from DistributedPython.ProbabilityOfCollision.Utils.get_covXcorr_parameters import get_covXcorr_parameters
from DistributedPython.ProbabilityOfCollision.Utils.TimeParabolaFit import TimeParabolaFit
from DistributedPython.ProbabilityOfCollision.Utils.PeakOverlapMD2 import PeakOverlapMD2
from DistributedPython.Utils.OrbitTransformations.orbit_period import orbit_period

def UsageViolationPc2D(r1, v1, C1, r2, v2, C2, HBR, params=None):
    """
    Calculate usage violation indicators for 2D-Pc conjunction plane method.

    Args:
        r1 (array-like): Primary object's position vector in ECI coordinates [m].
        v1 (array-like): Primary object's velocity vector in ECI coordinates [m/s].
        C1 (array-like): Primary's covariance matrix in ECI coordinate frame [m & s].
        r2 (array-like): Secondary object's position vector in ECI coordinates [m].
        v2 (array-like): Secondary object's velocity vector in ECI coordinates [m/s].
        C2 (array-like): Secondary's cov. matrix in ECI coordinate frame [m & s].
        HBR (float or array-like): Hard body radius [m].
        params (dict, optional): Parameter dictionary.

    Returns:
        tuple: (UVIndicators, out)
    """

    # Initializations and defaults
    if params is None:
        params = {}

    params = set_default_param(params, 'apply_covXcorr_corrections', True)
    params = set_default_param(params, 'remediate_NPD_TCA_eq_covariances', False)
    params = set_default_param(params, 'Fclip', 1e-4)
    params = set_default_param(params, 'ConjDurationGamma', 1e-16)
    params = set_default_param(params, 'verbose', 0)
    params = set_default_param(params, 'CPUVdebugging', 0)

    params = set_default_param(params, 'FullPc2DViolationAnalysis', True)
    params = set_default_param(params, 'AcceptableInitOffsetFactor', 10)
    params = set_default_param(params, 'TimeSigmaSpacing', 1)
    params = set_default_param(params, 'TimeSigmaDiffLimit', 2.5)
    params = set_default_param(params, 'QfunctionDiffLimit', 1.0)
    params = set_default_param(params, 'MaxRefinements', 20)

    params = set_default_param(params, 'RetrogradeReorientation', 1)

    UVIndicators = {
        'Inaccurate': np.nan,
        'NPDIssues': [np.nan, np.nan, np.nan],
        'Extended': np.nan,
        'Offset': np.nan
    }

    out = {
        'Nc2DSmallHBR': np.nan,
        'Nc2DLoLimit': np.nan,
        'Nc2DHiLimit': np.nan,
        'LogPcCorrectionFactor': np.nan,
        'params': params
    }

    r1 = np.asarray(r1).flatten()
    v1 = np.asarray(v1).flatten()
    r2 = np.asarray(r2).flatten()
    v2 = np.asarray(v2).flatten()
    C1 = np.asarray(C1)
    C2 = np.asarray(C2)

    if C1.shape != (6, 6):
        raise ValueError('C1 covariance must be a 6x6 matrix')
    if C2.shape != (6, 6):
        raise ValueError('C2 covariance must be a 6x6 matrix')

    # Process HBR
    if np.size(HBR) == 2:
        HBR = np.asarray(HBR)
        if np.any(HBR < 0):
             raise ValueError('Both HBR values must be nonnegative.')
        HBR = HBR[0] + HBR[1]
    elif np.size(HBR) != 1:
        raise ValueError('Input HBR must have one or two elements.')

    if HBR <= 0:
        raise ValueError('Combined HBR value must be positive.')

    out['HBR'] = HBR
    HBRkm = HBR / 1e3

    # Initialize parameters for PeakOverlapPos function
    POPPAR = {}
    POPPAR['verbose'] = params['verbose'] > 0
    POPPAR['Fclip'] = params['Fclip']
    POPPAR['maxiter'] = 100

    # Retrograde orbit processing
    if params['RetrogradeReorientation'] > 0:
        r1, v1, C1, r2, v2, C2, RRout = RetrogradeReorientation(r1, v1, C1, r2, v2, C2, params)
        out['RetrogradeReorientation'] = RRout['Reoriented']
    else:
        out['RetrogradeReorientation'] = False

    # Mean equinoctial matrices at nominal TCA
    out['Xmean10'], out['Pmean10'], out['Emean10'], out['Jmean10'], out['Kmean10'], \
    out['Qmean10'], out['Qmean10RemStat'], out['Qmean10Raw'], \
    out['Qmean10Rem'], out['C1Rem'] = EquinoctialMatrices(r1, v1, C1, \
    params['remediate_NPD_TCA_eq_covariances'])

    out['Xmean20'], out['Pmean20'], out['Emean20'], out['Jmean20'], out['Kmean20'], \
    out['Qmean20'], out['Qmean20RemStat'], out['Qmean20Raw'], \
    out['Qmean20Rem'], out['C2Rem'] = EquinoctialMatrices(r2, v2, C2, \
    params['remediate_NPD_TCA_eq_covariances'])

    # Check for undefined equinoctial elements
    # In Python, Emean10 is array. Check for nan.
    if np.any(np.isnan(out['Emean10'])) or np.any(np.isnan(out['Emean20'])):
        out['RefinementLevel'] = np.nan
        return UVIndicators, out

    # Get covariance cross correction parameters
    if params['apply_covXcorr_corrections']:
        XCprocessing, sigp, Gp, sigs, Gs = get_covXcorr_parameters(params)
        if XCprocessing:
            sigpXsigs = sigp * sigs
            Gp = Gp.T / 1e3 # Primary (6,1)
            Gs = Gs.T / 1e3 # Secondary (6,1)

            if out['RetrogradeReorientation']:
                M6 = RRout['M6']
                Gp = M6 @ Gp
                Gs = M6 @ Gs

            GEp = out['Kmean10'] @ Gp
            GEs = out['Kmean20'] @ Gs

            POPPAR['XCprocessing'] = True
            POPPAR['sigpXsigs'] = sigpXsigs
            POPPAR['GEp'] = GEp
            POPPAR['GEs'] = GEs
        else:
            POPPAR['XCprocessing'] = False
    else:
        XCprocessing = False
        POPPAR['XCprocessing'] = False

    out['covXcorr_corrections_applied'] = XCprocessing

    # Construct the nominal TCA relative state and covariance
    r = r2 - r1
    v = v2 - v1

    CRem = out['C1Rem'] + out['C2Rem']
    if XCprocessing:
        # Use Casali (2018) eq 11 & convert sensitivity vectors from km to m
        # Gs*Gp' + Gp*Gs'
        term = sigpXsigs * (np.outer(Gs, Gp) + np.outer(Gp, Gs)) * 1e6
        CRem = CRem - term

    # Rectilinear (RL) encounter Mahalanobis distances and Q functions
    A = CRem[0:3, 0:3]
    Lclip = (params['Fclip'] * HBR)**2

    Lraw, Veig = np.linalg.eigh(A)
    Leig = Lraw.copy()

    RelPosCovNPD = np.any(Leig <= 0)
    RelPosCovRemediation = Leig < Lclip
    Leig[RelPosCovRemediation] = Lclip

    Adet = np.prod(Leig)
    Ainv = (Veig * (1.0 / Leig)) @ Veig.T

    rT_Ai = r.T @ Ainv
    rT_Ai_r = rT_Ai @ r
    rT_Ai_v = rT_Ai @ v
    vT_Ai_v = v.T @ Ainv @ v

    if rT_Ai_v == 0:
        T = 0.0
    else:
        T = -rT_Ai_v / vT_Ai_v

    out['TRectilinear'] = T

    # 1-sigma Q function width in time for rectilinear encounter
    if vT_Ai_v == 0:
        SigmaSquaredRL = np.inf
    else:
        SigmaSquaredRL = 1.0 / vT_Ai_v
    out['STRectilinear'] = np.sqrt(SigmaSquaredRL)

    if params['TimeSigmaSpacing'] <= 0:
        raise ValueError('Invalid TimeSigmaSpacing parameter; recommendation = 1')

    dt = params['TimeSigmaSpacing'] * out['STRectilinear']
    t = np.array([T - dt, T, T + dt])
    MidPoint = 1 # 0-based index

    MD2Rectilinear = rT_Ai_r + t * (2 * rT_Ai_v) + (t**2) * vT_Ai_v
    logAdet = np.log(Adet / 1e18) # Account for km vs m units?
    # Wait, EquinoctialMatrices returns covariance in METERS.
    # C1, C2 inputs are meters.
    # But EquinoctialMatrices converts to km internally for equinoctial, but returns remediated C1Rem in...
    # Let's check EquinoctialMatrices.py return units.
    # "C1Rem/C2Rem = Remediated inertial Cartesian covariance at initial time (TCA) for the pri/sec".
    # Assuming meters as inputs were meters.
    # However, PeakOverlapMD2 uses HBRkm. And Ps is in km^2.
    # The MATLAB code divides Adet by 1e18. Adet is from A (CRem). CRem is in meters?
    # "C1 - Primary object's ECI covariance matrix (m position units)"
    # "C1Rem ... (m position units)"
    # So Adet is m^6. 1e18 is (1e3)^6. So converting to km^6.
    # Yes, PeakOverlapMD2 works in km units.

    out['tRectilinear'] = t
    out['QtRectilinear'] = MD2Rectilinear + logAdet
    out['MDtRectilinear'] = np.sqrt(MD2Rectilinear)
    out['QTRectilinear'] = out['QtRectilinear'][MidPoint]
    out['MDTRectilinear'] = out['MDtRectilinear'][MidPoint]
    out['VTRectilinear'] = np.linalg.norm(v / 1e3) # Convert to km/s

    # Define indicators for NPD covariance remediation
    Lraw1, _ = np.linalg.eigh(C1[0:3, 0:3])
    PriPosCovNPD = np.any(Lraw1 <= 0)

    Lraw2, _ = np.linalg.eigh(C2[0:3, 0:3])
    SecPosCovNPD = np.any(Lraw2 <= 0)

    UVIndicators['NPDIssues'] = [PriPosCovNPD, SecPosCovNPD, RelPosCovNPD]

    # Define initial usage violation indicators
    out['Qconverged'] = False
    out['TCurvilinear'] = np.nan
    out['QTCurvilinear'] = np.nan
    out['STCurvilinear'] = np.nan
    out['VTCurvilinear'] = np.nan

    sqrt2_erfcinv_gamma = np.sqrt(2) * erfcinv(params['ConjDurationGamma'])
    dtau = out['STRectilinear'] * sqrt2_erfcinv_gamma
    out['Ta'] = out['TRectilinear'] - dtau
    out['Tb'] = out['TRectilinear'] + dtau

    out['Period1'] = orbit_period(r1, v1)
    out['Period2'] = orbit_period(r2, v2)
    out['PeriodMin'] = min(out['Period1'], out['Period2'])

    UVIndicators['Extended'] = min(1.0, 2 * dtau / out['PeriodMin'])

    Tab = max(abs(out['Ta']), abs(out['Tb']))
    UVIndicators['Offset'] = min(1.0, Tab / out['PeriodMin'])

    if not params['FullPc2DViolationAnalysis']:
        out['RefinementLevel'] = -2
        return UVIndicators, out

    if np.isinf(out['STRectilinear']):
        out['RefinementLevel'] = -1
        return UVIndicators, out

    # Curvilinear encounter Mahalanobis distances and Q functions
    out['RefinementLevel'] = 0
    out['tCurvilinear'] = t
    out['QtCurvilinear'] = np.full(t.shape, np.nan)

    Xu = None
    Ps = None

    for nt in range(3):
        QnewCL, XXu, PPs, _, _, _, _ = PeakOverlapMD2(t[nt], \
            0, out['Emean10'], out['Qmean10'], \
            0, out['Emean20'], out['Qmean20'], \
            HBRkm, 1, POPPAR)
        out['QtCurvilinear'][nt] = QnewCL
        if nt == MidPoint:
            Xu = XXu
            Ps = PPs

    Qconverged = not np.any(np.isnan(out['QtCurvilinear']))

    if Qconverged:
        out['RefinementLevel'] = 1

        Qdotdot = (out['QtCurvilinear'][2] - 2 * out['QtCurvilinear'][1] + out['QtCurvilinear'][0]) / (dt**2)

        if Qdotdot <= 0:
            out['STCurvilinear'] = np.inf
        else:
            res_tpf = TimeParabolaFit(out['tCurvilinear'], out['QtCurvilinear'])
            abc = res_tpf['c']

            t0 = -abc[1] / Qdotdot
            Qt0 = abc[2] - abc[1]**2 / abc[0] / 4.0

            AcceptableInitOffset = True
            delt0max = params['AcceptableInitOffsetFactor'] * (out['tCurvilinear'][2] - out['tCurvilinear'][0])

            if t0 < out['tCurvilinear'][0] - delt0max:
                AcceptableInitOffset = False
                t0 = out['tCurvilinear'][0] - delt0max
                Qt0 = abc[0]*t0**2 + abc[1]*t0 + abc[2]
            elif t0 > out['tCurvilinear'][2] + delt0max:
                AcceptableInitOffset = False
                t0 = out['tCurvilinear'][2] + delt0max
                Qt0 = abc[0]*t0**2 + abc[1]*t0 + abc[2]

            out['TCurvilinear'] = t0
            out['QTCurvilinear'] = Qt0
            SigmaSquaredCL = 2.0 / Qdotdot
            out['STCurvilinear'] = np.sqrt(SigmaSquaredCL)

            out['VTCurvilinear'] = np.linalg.norm(Xu[3:6])

            out['LogPcCorrectionFactor'] = \
                np.log(out['STCurvilinear'] / out['STRectilinear']) \
                + np.log(out['VTCurvilinear'] / out['VTRectilinear']) \
                + (out['QTRectilinear'] - out['QTCurvilinear']) / 2.0

            if AcceptableInitOffset:
                TimeDifference = out['TCurvilinear'] - out['TRectilinear']
                MaxTimeDifference = params['TimeSigmaDiffLimit'] * min(out['STRectilinear'], out['STCurvilinear'])
                QDifference = out['QTCurvilinear'] - out['QTRectilinear']

                NeedsRefinement = (abs(TimeDifference) >= MaxTimeDifference) or \
                                  (abs(QDifference) >= params['QfunctionDiffLimit'])
            else:
                NeedsRefinement = True

            while NeedsRefinement:
                out['RefinementLevel'] += 1

                ItMinimum = np.argmin(out['QtCurvilinear'])

                tnewCurvilinear = out['TCurvilinear']
                QnewCurvilinear, Xu, Ps, _, _, _, _ = PeakOverlapMD2( \
                    tnewCurvilinear, \
                    0, out['Emean10'], out['Qmean10'], \
                    0, out['Emean20'], out['Qmean20'], \
                    HBRkm, 1, POPPAR)

                if np.isnan(QnewCurvilinear):
                    Qconverged = False

                    min_tCurvilinear = np.min(out['tCurvilinear'])
                    max_tCurvilinear = np.max(out['tCurvilinear'])

                    if out['TCurvilinear'] < min_tCurvilinear:
                        tnew = (out['TCurvilinear'] + min_tCurvilinear) / 2.0
                    elif out['TCurvilinear'] > max_tCurvilinear:
                        tnew = (out['TCurvilinear'] + max_tCurvilinear) / 2.0
                    else:
                        tnew = np.nan

                    if np.isnan(tnew) or len(out['tCurvilinear']) > 3:
                        NeedsRefinement = False
                    else:
                        if out['RefinementLevel'] < params['MaxRefinements']:
                            NeedsRefinement = True
                            out['TCurvilinear'] = tnew
                            out['QTCurvilinear'] = np.nan
                        else:
                            NeedsRefinement = False
                else:
                    Qconverged = True

                    Qsrt = np.sort(out['QtCurvilinear'])
                    Qcut = Qsrt[2] # 3rd element
                    Qdecreasing = QnewCurvilinear < Qcut

                    out['tCurvilinear'] = np.append(out['tCurvilinear'], tnewCurvilinear)
                    out['QtCurvilinear'] = np.append(out['QtCurvilinear'], QnewCurvilinear)

                    if Qdecreasing:
                        out['VTCurvilinear'] = np.linalg.norm(Xu[3:6])

                        res_tpf = TimeParabolaFit(out['tCurvilinear'], out['QtCurvilinear'])
                        yy = res_tpf['c']

                        if yy[0] <= 0:
                            out['STCurvilinear'] = np.inf
                        else:
                            out['TCurvilinear'] = -yy[1] / yy[0] / 2.0
                            out['QTCurvilinear'] = yy[2] - yy[1]**2 / yy[0] / 4.0
                            SigmaSquaredCL = 1.0 / yy[0]
                            out['STCurvilinear'] = np.sqrt(SigmaSquaredCL)
                    else:
                        NtC = len(out['tCurvilinear']) - 1 # Exclude latest
                        min_tCurvilinear = np.min(out['tCurvilinear'][:NtC])
                        max_tCurvilinear = np.max(out['tCurvilinear'][:NtC])

                        if out['TCurvilinear'] < min_tCurvilinear:
                            tnew = (out['TCurvilinear'] + min_tCurvilinear) / 2.0
                            res_tpf = TimeParabolaFit(out['tCurvilinear'], out['QtCurvilinear'])
                            yy = res_tpf['c']
                            Qnew = yy[0]*tnew**2 + yy[1]*tnew + yy[2]

                        elif out['TCurvilinear'] > max_tCurvilinear:
                            tnew = (out['TCurvilinear'] + max_tCurvilinear) / 2.0
                            res_tpf = TimeParabolaFit(out['tCurvilinear'], out['QtCurvilinear'])
                            yy = res_tpf['c']
                            Qnew = yy[0]*tnew**2 + yy[1]*tnew + yy[2]
                        else:
                            tnew = out['tCurvilinear'][ItMinimum]
                            Qnew = out['QtCurvilinear'][ItMinimum]

                        out['TCurvilinear'] = tnew
                        out['QTCurvilinear'] = Qnew

                if not np.isinf(out['STCurvilinear']):
                     out['LogPcCorrectionFactor'] = \
                        np.log(out['STCurvilinear'] / out['STRectilinear']) \
                        + np.log(out['VTCurvilinear'] / out['VTRectilinear']) \
                        + (out['QTRectilinear'] - out['QTCurvilinear']) / 2.0

                NeedsRefinement = False
                if not np.isinf(out['STCurvilinear']):
                    if out['RefinementLevel'] < params['MaxRefinements']:
                         TimeDifference = out['TCurvilinear'] - out['tCurvilinear'][ItMinimum]
                         MaxTimeDifference = params['TimeSigmaDiffLimit'] * out['STCurvilinear']
                         QDifference = out['QTCurvilinear'] - out['QtCurvilinear'][ItMinimum]

                         NeedsRefinement = (abs(TimeDifference) >= MaxTimeDifference) or \
                                           (abs(QDifference) >= params['QfunctionDiffLimit'])

    if np.isinf(out['STCurvilinear']) or np.isnan(out['STCurvilinear']):
        Qconverged = False

    if Qconverged:
        tnewCurvilinear = out['TCurvilinear']
        QnewCurvilinear, Xu, _, Asdet, Asinv, POPconv, AsAux = PeakOverlapMD2( \
            tnewCurvilinear, \
            0, out['Emean10'], out['Qmean10'], \
            0, out['Emean20'], out['Qmean20'], \
            HBRkm, 1, POPPAR)

        if not POPconv:
            Qconverged = False
        else:
            out['QTCurvilinear'] = QnewCurvilinear
            out['LogPcCorrectionFactor'] = \
                np.log(out['STCurvilinear'] / out['STRectilinear']) \
                + np.log(out['VTCurvilinear'] / out['VTRectilinear']) \
                + (out['QTRectilinear'] - out['QTCurvilinear']) / 2.0

            HBR2 = HBRkm**2
            Nc2DCoef = HBR2 * out['VTCurvilinear'] * out['STCurvilinear'] / 2.0
            out['Nc2DSmallHBR'] = min(1.0, Nc2DCoef * np.exp(-out['QTCurvilinear'] / 2.0))

            Asiru = Asinv @ Xu[0:3]
            Aterm = 2 * HBRkm * np.sqrt(Asiru.T @ Asiru)
            QTmax = HBR2 / np.min(AsAux['AsLeig']) + Aterm + out['QTCurvilinear']
            out['Nc2DLoLimit'] = min(1.0, Nc2DCoef * np.exp(-QTmax / 2.0))

            logAsdet = np.log(Asdet)
            MD2Curvilinear = out['QTCurvilinear'] - logAsdet
            MD2min = HBR2 / np.max(AsAux['AsLeig']) - Aterm + MD2Curvilinear
            MD2min = max(0.0, MD2min)
            QTmin = MD2min + logAsdet
            out['Nc2DHiLimit'] = min(1.0, Nc2DCoef * np.exp(-QTmin / 2.0))

    out['Qconverged'] = Qconverged

    if Qconverged:
        if np.isinf(out['STCurvilinear']):
            UVIndicators['Extended'] = 1.0
        else:
            dtau = out['STCurvilinear'] * sqrt2_erfcinv_gamma
            out['Ta'] = out['TCurvilinear'] - dtau
            out['Tb'] = out['TCurvilinear'] + dtau

            UVIndicators['Extended'] = min(1.0, (out['Tb'] - out['Ta']) / out['PeriodMin'])

            Tab = max(abs(out['Ta']), abs(out['Tb']))
            UVIndicators['Offset'] = min(1.0, Tab / out['PeriodMin'])

            UVIndicators['Inaccurate'] = 1.0 - np.exp(-abs(out['LogPcCorrectionFactor']))

    return UVIndicators, out
