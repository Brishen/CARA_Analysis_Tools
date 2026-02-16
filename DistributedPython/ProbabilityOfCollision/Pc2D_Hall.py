import numpy as np
import warnings
from scipy.special import erfc
from scipy.integrate import dblquad
from scipy.optimize import minimize
from DistributedPython.ProbabilityOfCollision.Utils.set_default_param import set_default_param
from DistributedPython.ProbabilityOfCollision.Utils.RetrogradeReorientation import RetrogradeReorientation
from DistributedPython.ProbabilityOfCollision.Utils.EquinoctialMatrices import EquinoctialMatrices
from DistributedPython.ProbabilityOfCollision.Utils.get_covXcorr_parameters import get_covXcorr_parameters
from DistributedPython.ProbabilityOfCollision.Utils.UsageViolationPc2D import UsageViolationPc2D
from DistributedPython.ProbabilityOfCollision.Utils.PeakOverlapMD2 import PeakOverlapMD2
from DistributedPython.ProbabilityOfCollision.Utils.CovRemEigValClip import CovRemEigValClip
from DistributedPython.Utils.AugmentedMath.shift_bisect_minsearch import shift_bisect_minsearch
from DistributedPython.ProbabilityOfCollision.Utils.MD2MinRefine import MD2MinRefine
from DistributedPython.ProbabilityOfCollision.PcCircle import PcCircle
from DistributedPython.ProbabilityOfCollision.Utils.getLebedevSphere import getLebedevSphere

def Pc2D_Hall(r1, v1, C1, r2, v2, C2, HBR, params=None):
    """
    Calculate a single-conjunction Pc using the 2D-Nc algorithm.

    Args:
        r1 (array-like): Primary object's ECI position vector (m).
        v1 (array-like): Primary object's ECI velocity vector (m/s).
        C1 (array-like): Primary object's ECI covariance matrix (m position units).
        r2 (array-like): Secondary object's ECI position vector (m).
        v2 (array-like): Secondary object's ECI velocity vector (m/s).
        C2 (array-like): Secondary object's ECI covariance matrix (m position units).
        HBR (float or array-like): Combined primary+secondary hard-body radii.
        params (dict, optional): Auxiliary input parameter structure.

    Returns:
        tuple: (Pc, out)
    """

    # Initializations and defaults
    if params is None:
        params = {}

    params = set_default_param(params, 'CalcConjTimes', True)
    params = set_default_param(params, 'apply_covXcorr_corrections', True)
    params = set_default_param(params, 'remediate_NPD_TCA_eq_covariances', False)
    params = set_default_param(params, 'ForceQuad2dPcCutoff', 5e-5)

    params = set_default_param(params, 'Log10Quad2dPc', np.array([-13, -7, -5, -4]))
    params = set_default_param(params, 'Log10Quad2dRelTol', np.array([-3, -4, -5, -6]))

    if len(params['Log10Quad2dPc']) != len(params['Log10Quad2dRelTol']) or \
       np.any(np.diff(params['Log10Quad2dPc']) <= 0) or \
       np.any(np.diff(params['Log10Quad2dRelTol']) >= 0):
        raise ValueError('Invalid Log10Quad2dPc & Log10Quad2dRelTol table parameters')

    params = set_default_param(params, 'RelDifConjPlane', 1e-2)
    params = set_default_param(params, 'Fclip', 1e-4)
    params = set_default_param(params, 'deg_Lebedev', 5810)
    params = set_default_param(params, 'Pc_tiny', 1e-300)
    params = set_default_param(params, 'verbose', 0)
    params = set_default_param(params, 'RetrogradeReorientation', 1)
    params = set_default_param(params, 'SlowMode', False)

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

    out = {}
    out['PcUS'] = None; out['PcLeb'] = None; out['PcAlt'] = None; out['PcCP'] = None; out['Pcmethod'] = np.nan

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

    POPPAR = {}
    POPPAR['verbose'] = params['verbose'] > 0
    POPPAR['Fclip'] = params['Fclip']
    POPPAR['maxiter'] = 100

    if np.isinf(HBR):
        Pc = 1.0
        return Pc, out

    # Retrograde orbit processing
    if params['RetrogradeReorientation'] > 0:
        r1, v1, C1, r2, v2, C2, RRout = RetrogradeReorientation(r1, v1, C1, r2, v2, C2, params)
        out['RetrogradeReorientation'] = RRout['Reoriented']
    else:
        out['RetrogradeReorientation'] = False

    if out['RetrogradeReorientation']:
        params['UVPc2D'] = []

    # Mean equinoctial matrices at nominal TCA for primary and secondary
    UVPc2D_Supplied = False
    Use_UVPc2D_Matrices = False

    if params.get('UVPc2D') is not None and len(params['UVPc2D']) > 0:
        UVPc2D_Supplied = True
        # Check remediate_NPD_TCA_eq_covariances consistency
        # UVPc2D structure has 'params' field? UsageViolationPc2D returns 'out' which has 'params'.
        # Assuming params['UVPc2D'] is the 'out' structure from UsageViolationPc2D.
        uv_params = params['UVPc2D'].get('params', {})
        if uv_params.get('remediate_NPD_TCA_eq_covariances') == params['remediate_NPD_TCA_eq_covariances']:
            Use_UVPc2D_Matrices = True

    if Use_UVPc2D_Matrices:
        uv = params['UVPc2D']
        out['Xmean10'] = uv['Xmean10']; out['Pmean10'] = uv['Pmean10']
        out['Emean10'] = uv['Emean10']; out['Jmean10'] = uv['Jmean10']; out['Kmean10'] = uv['Kmean10']
        out['Qmean10'] = uv['Qmean10']; out['Qmean10RemStat'] = uv['Qmean10RemStat']
        out['Qmean10Raw'] = uv['Qmean10Raw']; out['Qmean10Rem'] = uv['Qmean10Rem']; out['C1Rem'] = uv['C1Rem']

        out['Xmean20'] = uv['Xmean20']; out['Pmean20'] = uv['Pmean20']
        out['Emean20'] = uv['Emean20']; out['Jmean20'] = uv['Jmean20']; out['Kmean20'] = uv['Kmean20']
        out['Qmean20'] = uv['Qmean20']; out['Qmean20RemStat'] = uv['Qmean20RemStat']
        out['Qmean20Raw'] = uv['Qmean20Raw']; out['Qmean20Rem'] = uv['Qmean20Rem']; out['C2Rem'] = uv['C2Rem']
    else:
        out['Xmean10'], out['Pmean10'], out['Emean10'], out['Jmean10'], out['Kmean10'], \
        out['Qmean10'], out['Qmean10RemStat'], out['Qmean10Raw'], \
        out['Qmean10Rem'], out['C1Rem'] = EquinoctialMatrices(r1, v1, C1, \
        params['remediate_NPD_TCA_eq_covariances'])

        out['Xmean20'], out['Pmean20'], out['Emean20'], out['Jmean20'], out['Kmean20'], \
        out['Qmean20'], out['Qmean20RemStat'], out['Qmean20Raw'], \
        out['Qmean20Rem'], out['C2Rem'] = EquinoctialMatrices(r2, v2, C2, \
        params['remediate_NPD_TCA_eq_covariances'])

    if np.any(np.isnan(out['Emean10'])) or np.any(np.isnan(out['Emean20'])):
        Pc = np.nan
        return Pc, out

    # Get covariance cross correlation parameters
    XCprocessing = False
    POPPAR['XCprocessing'] = False

    if params['apply_covXcorr_corrections']:
        XCprocessing, sigp, Gp, sigs, Gs = get_covXcorr_parameters(params)
        if XCprocessing:
            sigpXsigs = sigp * sigs
            Gp = Gp.T / 1e3 # (6,1)
            Gs = Gs.T / 1e3 # (6,1)

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

    out['covXcorr_corrections_applied'] = XCprocessing

    # Construct the nominal TCA relative state and covariance
    r = r2 - r1
    v = v2 - v1

    CRem = out['C1Rem'] + out['C2Rem']
    if XCprocessing:
        term = sigpXsigs * (np.outer(Gs, Gp) + np.outer(Gp, Gs)) * 1e6
        CRem = CRem - term

    # Run UsageViolationPc2D
    UVPc2D = {}
    if UVPc2D_Supplied:
        if params['UVPc2D']['covXcorr_corrections_applied'] == out['covXcorr_corrections_applied']:
            UVPc2D = params['UVPc2D']

    if not UVPc2D:
        _, UVPc2D = UsageViolationPc2D(r1, v1, C1, r2, v2, C2, HBR, params)

    # Find the nearby time of minimum effective Mahalnobis distance
    out['NeedFullMD2Search'] = True

    # Define EMD2fun
    EMD2fun = lambda tt: PeakOverlapMD2(tt, \
                                   0, out['Emean10'], out['Qmean10'], \
                                   0, out['Emean20'], out['Qmean20'], \
                                   HBRkm, 1, POPPAR)[0] # PeakOverlapMD2 returns tuple, [0] is MD2

    if UVPc2D and 'STCurvilinear' in UVPc2D and \
       not np.isnan(UVPc2D['STCurvilinear']) and not np.isinf(UVPc2D['STCurvilinear']):

       xtol = 1e-2 * UVPc2D['STCurvilinear']
       ytol = 1e-3

       tprev = UVPc2D['tCurvilinear']
       yprev = UVPc2D['QtCurvilinear']

       # unique
       tprev, unique_ind = np.unique(tprev, return_index=True)
       yprev = yprev[unique_ind]
       Nprev = tprev.size

       ipmin = np.argmin(yprev)

       tgrd = np.array([])
       ygrd = np.array([])
       cgrd = np.array([], dtype=bool)

       if ipmin != 0 and ipmin != Nprev - 1:
           # Bracketed
           jpmin = ipmin - 1
           kpmin = ipmin + 1
           tdel = min(tprev[ipmin] - tprev[jpmin], tprev[kpmin] - tprev[ipmin]) / 2.0

           Ngrd = 5
           tgrd = np.array([tprev[jpmin], tprev[ipmin]-tdel, tprev[ipmin], tprev[ipmin]+tdel, tprev[kpmin]])
           ygrd = np.array([yprev[jpmin], np.nan, yprev[ipmin], np.nan, yprev[kpmin]])
           cgrd = np.isnan(ygrd)

           Nbisectmax = 100
           Nshiftmax = 100

       else:
           Nsig0 = abs(UVPc2D['TCurvilinear']) / UVPc2D['STCurvilinear']
           Nsig = min(max(1.0, Nsig0), 10.0)

           Ts = UVPc2D['TCurvilinear']
           Ws = Nsig * UVPc2D['STCurvilinear']
           Nbisectmax = 100
           Nshiftmax = max(100, 3 * round((Nsig0 + 1) / Nsig))

           Ngrd = 5
           halfWs = 0.5 * Ws
           tgrd = np.array([Ts, Ts-halfWs, Ts+halfWs, Ts-Ws, Ts+Ws])
           ygrd = np.zeros(tgrd.shape)
           cgrd = np.ones(tgrd.shape, dtype=bool)

           for ngrd in range(Ngrd):
               dist = np.abs(tgrd[ngrd] - tprev)
               icls = np.argmin(dist)
               tcls = dist[icls]

               if tcls < halfWs:
                   tgrd[ngrd] = tprev[icls]
                   ygrd[ngrd] = yprev[icls]
                   cgrd[ngrd] = False

                   mask = np.ones(tprev.shape, dtype=bool)
                   mask[icls] = False
                   tprev = tprev[mask]
                   yprev = yprev[mask]

                   if tprev.size == 0:
                       break

       for ngrd in range(Ngrd):
           if cgrd[ngrd]:
               ygrd[ngrd] = EMD2fun(tgrd[ngrd])

           if np.isnan(ygrd[ngrd]):
               break

       tgrid = np.concatenate((tgrd, UVPc2D['tCurvilinear']))
       ygrid = np.concatenate((ygrd, UVPc2D['QtCurvilinear']))

       if not np.any(np.isnan(ygrid)):
           srt = np.argsort(tgrid)
           tgrid = tgrid[srt]
           ygrid = ygrid[srt]

           xmin, ymin, SearchConverged, _, _, _, _ = shift_bisect_minsearch(
               EMD2fun, tgrid, ygrid, Nbisectmax, Nshiftmax, xtol, ytol, 0
           )

           if SearchConverged:
               out['NeedFullMD2Search'] = False
               out['MD2SearchConvergence'] = True
               findmin = not UVPc2D['Qconverged']
               tmin = xmin
               tsg = UVPc2D['STCurvilinear']

    if out['NeedFullMD2Search']:

        T = 0.0
        iterating = True
        converged = False
        iter_count = 0
        itermin = 2
        itermax = 25
        xtol = 1e-1

        while iterating:
            iter_count += 1
            M0T, Xu, _, _, Asinv, POPconv, _ = PeakOverlapMD2(
                T, 0, out['Emean10'], out['Qmean10'], 0, out['Emean20'], out['Qmean20'], HBRkm, 0, POPPAR
            )

            if np.isnan(M0T):
                iterating = False
            else:
                ru = Xu[0:3]
                vu = Xu[3:6]
                vuTAsinv = vu.T @ Asinv
                acoef = vuTAsinv @ vu
                bhalf = vuTAsinv @ ru

                dT = -bhalf / acoef
                sT = np.sqrt(1.0 / acoef)

                if np.isnan(dT):
                    iterating = False
                    converged = False
                elif abs(dT) < xtol * sT and iter_count >= itermin:
                    iterating = False
                    converged = POPconv
                elif iter_count >= itermax:
                    iterating = False
                else:
                    T = T + dT

        out['MD2SearchConvergence'] = converged

        if converged:
            tmin = T
            tsg = sT
            findmin = True
            out['MD2SearchConvergence'] = converged
        else:
            Lclip = (params['Fclip'] * HBR)**2
            res_cov = CovRemEigValClip(CRem[0:3, 0:3], Lclip)
            Ainv = res_cov['Ainv']

            vtAinv = v.T @ Ainv
            acoef = vtAinv @ v
            bhalf = vtAinv @ r
            ccoef = r.T @ Ainv @ r

            out['tMDmin0'] = -bhalf / acoef
            out['MDmin0'] = np.sqrt(ccoef - bhalf**2 / acoef)
            out['SigmaT0'] = np.sqrt(1.0 / acoef)

            if acoef == 0:
                Pc = np.nan
                out['Pcmethod'] = 0
                return Pc, out

            xtol = 1e-2 * out['SigmaT0']
            ytol = 1e-3

            Nsig0 = abs(out['tMDmin0']) / out['SigmaT0']
            Nsig = min(max(1.0, Nsig0), 10.0)

            Ts = 0.0
            Ws = Nsig * out['SigmaT0']
            Nbisectmax = 100
            Nshiftmax = max(100, 3 * round((Nsig0 + 1) / Nsig))

            tgrid = np.array([Ts - 1.0*Ws, Ts - 0.5*Ws, Ts, Ts + 0.5*Ws, Ts + 1.0*Ws])
            ygrid = np.zeros(tgrid.shape)

            valid = True
            for ngrid in range(len(ygrid)):
                ygrid[ngrid] = EMD2fun(tgrid[ngrid])
                if np.isnan(ygrid[ngrid]):
                    Pc = np.nan
                    out['Pcmethod'] = 0
                    valid = False
                    break

            if not valid:
                return Pc, out

            xmin, ymin, PrimaryConverged, _, _, _, ybuf = shift_bisect_minsearch(
                EMD2fun, tgrid, ygrid, Nbisectmax, Nshiftmax, xtol, ytol, 0
            )

            if PrimaryConverged:
                tmin = xmin
                findmin = False
                out['MD2SearchConvergence'] = 2
            else:
                if np.any(np.isnan(ybuf)):
                    Pc = np.nan
                    out['Pcmethod'] = 0
                    return Pc, out

                tmin = xmin
                res_min = minimize(EMD2fun, tmin, method='Nelder-Mead', tol=xtol, options={'maxfev': 2000, 'xatol': xtol, 'fatol': ytol, 'disp': False})

                BackupConverged = res_min.success
                xmin = res_min.x[0]
                ymin = res_min.fun

                if BackupConverged:
                    out['MD2SearchConvergence'] = 3
                    tmin = xmin
                    findmin = False
                else:
                    tmin = out['tMDmin0']
                    findmin = True
                    if params['verbose']:
                        print('MD2MinRefine minimization method being used')

            tsg = out['SigmaT0']

    deltsg = 1
    ttol = 1e-3
    itermax = 15

    MS = MD2MinRefine(tmin, tsg, deltsg, ttol, itermax, findmin, \
        out['Emean10'], out['Qmean10'], out['Emean20'], out['Qmean20'], HBRkm, POPPAR)

    if not MS['MD2converged']:
        ttol = 3e-3
        itermax = 20
        MS = MD2MinRefine(tmin, tsg, deltsg, ttol, itermax, findmin, \
            out['Emean10'], out['Qmean10'], out['Emean20'], out['Qmean20'], HBRkm, POPPAR)

    if not MS['MD2converged']:
        Pc = np.nan
        out['Pcmethod'] = 0
        return Pc, out
    else:
        if out.get('MD2SearchConvergence', 0) == 0:
            out['MD2SearchConvergence'] = 4

    tmin = MS['tmd']
    ymin = MS['ymd']
    Xu = MS['Xu']
    ru = Xu[0:3]
    vu = Xu[3:6]
    Asinv = MS['Asinv']
    Asdet = MS['Asdet']
    logAsdet = np.log(Asdet)
    Ps = MS['Ps']
    delt = MS['delt']
    twodelt = 2 * delt
    delt2 = delt**2

    rudot = (MS['Xuhi'][0:3] - MS['Xulo'][0:3]) / twodelt
    rudotdot = (MS['Xuhi'][0:3] - 2*ru + MS['Xulo'][0:3]) / delt2

    Asinvdot = (MS['Asinvhi'] - MS['Asinvlo']) / twodelt
    Asinvdotdot = (MS['Asinvhi'] - 2*Asinv + MS['Asinvlo']) / delt2

    Q0T = ymin
    Q0Tdot = MS['ydot']
    Q0Tdotdot = MS['ydotdot']

    out['TQ0min'] = tmin - Q0Tdot / Q0Tdotdot
    out['Q0min'] = Q0T - Q0Tdot**2 / Q0Tdotdot / 2.0
    out['SigmaQ0min'] = np.sqrt(2.0 / Q0Tdotdot)

    out['HBRTime'] = HBRkm / np.linalg.norm(rudot)
    if out['SigmaQ0min'] == out['HBRTime']:
        out['HBRTimeRatio'] = 1.0
    else:
        out['HBRTimeRatio'] = out['HBRTime'] / out['SigmaQ0min']

    # As = Ps[0:3, 0:3]
    Bs = Ps[3:6, 0:3]
    Cs = Ps[3:6, 3:6]
    bs = Bs @ Asinv
    Csp = Cs - bs @ Bs.T

    # Conjunction plane estimation mode
    out['covCAeff'] = Ps[0:3, 0:3] * 1e6
    vup = vu - bs @ ru
    out['vCAeff'] = vup * 1e3

    out['TCAeff'] = -(ru.T @ vup) / (vup.T @ vup)
    out['rCAeff'] = (ru + vup * out['TCAeff']) * 1e3

    z1x3 = np.zeros((1, 3))
    rCAeff_reshaped = out['rCAeff'].reshape(1, 3)
    vCAeff_reshaped = out['vCAeff'].reshape(1, 3)
    covCAeff_reshaped = out['covCAeff'].reshape(1, 3, 3)

    out['PcCP'], out['PcCPInfo'] = PcCircle(z1x3, z1x3, np.zeros((1, 3, 3)), \
        rCAeff_reshaped, vCAeff_reshaped, covCAeff_reshaped, HBR)

    # Unit sphere estimation mode
    R = HBRkm

    deg_Lebedev = params['deg_Lebedev']
    sph_Lebedev = getLebedevSphere(deg_Lebedev)
    vec_Lebedev = np.column_stack((sph_Lebedev.x, sph_Lebedev.y, sph_Lebedev.z)).T # (3, N)
    wgt_Lebedev = sph_Lebedev.w # (N,)

    integ, RUI = runit_integrand(vec_Lebedev, R, ru, vu, bs, Csp, Asinv, Asinvdot, Asinvdotdot, logAsdet, \
        rudot, rudotdot, Q0T, Q0Tdot, Q0Tdotdot, params['SlowMode'])

    out['MRstarNegativesB'] = RUI['MRstarNegatives']
    out['SRstarImaginariesB'] = RUI['SRstarImaginaries']

    Sum0Vec = wgt_Lebedev * integ
    Sum0 = np.sum(Sum0Vec)
    PcConst = R**2 / np.sqrt(2 * np.pi)**3
    out['PcLeb'] = PcConst * Sum0

    if params['CalcConjTimes']:
        Tstar = tmin - RUI['QRTdot'] / RUI['QRTdotdot']
        Sum1Vec = Sum0Vec * Tstar
        Sum2Vec = Sum0Vec * (Tstar**2 + RUI['SRstar2'])
        Sum1 = np.sum(Sum1Vec)
        Sum2 = np.sum(Sum2Vec)
        out['TMeanRate'] = Sum1 / Sum0
        out['TSigmaRate'] = np.sqrt((Sum2 / Sum0) - out['TMeanRate']**2)

    if np.isnan(out['PcLeb']):
        need_quad2d = False
        out['PcUS'] = np.nan
        out['PcAlt'] = out['PcCP']
        Pc = np.nan
        out['Pcmethod'] = 0

    elif out['PcLeb'] > params['Pc_tiny']:
        NumVec = np.sum(Sum0Vec > 1e-3 * np.max(Sum0Vec))
        out['LebNumb'] = NumVec
        NumCut = 0.025 * deg_Lebedev

        if NumVec <= NumCut:
            if out['PcCPInfo']['ClipBoundSet']:
                need_quad2d = False
                out['PcUS'] = out['PcLeb']
                out['PcAlt'] = out['PcCP']
                Pc = min(1.0, out['PcCP'])
                out['Pcmethod'] = 1
            else:
                need_quad2d = True
        elif out['PcLeb'] >= params['ForceQuad2dPcCutoff']:
            need_quad2d = True
        else:
            if np.isnan(out['PcCP']) or abs(out['PcCP'] - out['PcLeb']) > params['RelDifConjPlane'] * out['PcLeb']:
                need_quad2d = True
            else:
                need_quad2d = False
                out['PcUS'] = out['PcLeb']
                out['PcAlt'] = out['PcCP']
                Pc = min(1.0, out['PcUS'])
                out['Pcmethod'] = 2
    else:
        need_quad2d = False
        out['PcUS'] = out['PcLeb']
        out['PcAlt'] = out['PcLeb']
        Pc = min(1.0, out['PcUS'])
        out['Pcmethod'] = 2

    if need_quad2d:
        Log10PcLeb = np.log10(out['PcLeb'])
        if Log10PcLeb <= params['Log10Quad2dPc'][0]:
            Log10RelTolQuad2d = params['Log10Quad2dRelTol'][0]
        elif Log10PcLeb >= params['Log10Quad2dPc'][-1]:
            Log10RelTolQuad2d = params['Log10Quad2dRelTol'][-1]
        else:
            Log10RelTolQuad2d = np.interp(Log10PcLeb, params['Log10Quad2dPc'], params['Log10Quad2dRelTol'])

        RelTolQuad2d = 10.0**Log10RelTolQuad2d
        AbsTolQuad2d = RelTolQuad2d * out['PcLeb']
        if not np.isnan(out['PcCP']):
            AbsTolQuad2d = min(AbsTolQuad2d, RelTolQuad2d * out['PcCP'])

        fun = lambda tht, phi: Nc2D_integrand(phi, tht, \
            R, ru, vu, bs, Csp, Asinv, \
            Asinvdot, Asinvdotdot, logAsdet, \
            rudot, rudotdot, Q0T, Q0Tdot, Q0Tdotdot, \
            params['SlowMode'])

        # quad2d(fun,0,2*pi,0,pi,...)
        # dblquad(func, 0, 2*pi, lambda x: 0, lambda x: pi)
        # func order: (y, x). Here y=tht, x=phi.
        integ, error = dblquad(fun, 0, 2*np.pi, lambda x: 0, lambda x: np.pi, epsabs=AbsTolQuad2d, epsrel=RelTolQuad2d)

        out['PcUS'] = PcConst * integ

        if out['PcUS'] == out['PcCP']:
            difCP = 0
        else:
            difCP = abs(out['PcUS'] - out['PcCP']) / ((out['PcUS'] + out['PcCP']) / 2.0)

        if out['PcUS'] == out['PcLeb']:
            difLeb = 0
        else:
            difLeb = abs(out['PcUS'] - out['PcLeb']) / ((out['PcUS'] + out['PcLeb']) / 2.0)

        if difCP > difLeb:
            out['PcAlt'] = out['PcCP']
        else:
            out['PcAlt'] = out['PcLeb']

        if (np.isnan(out['PcUS']) or out['PcUS'] > 1) and not np.isnan(out['PcCP']):
            out['PcAlt'] = out['PcUS']
            Pc = out['PcCP']
            out['Pcmethod'] = 1
        else:
            Pc = out['PcUS']
            out['Pcmethod'] = 3

        if not np.isnan(Pc):
            Pc = min(1.0, Pc)

    return Pc, out

def Nc2D_integrand(phi, tht, R, ru, vu, bs, Csp, Asinv, Asinvdot, Asinvdotdot, logAsdet, rudot, rudotdot, Q0T, Q0Tdot, Q0Tdotdot, SlowMode):
    # phi and tht can be scalars or arrays
    # In dblquad they are scalars.
    # If using vectorized integration (not dblquad), they might be arrays.

    phi = np.atleast_1d(phi)
    tht = np.atleast_1d(tht)

    # Flatten
    phi = phi.flatten()
    tht = tht.flatten()
    N = phi.size

    rhat = np.full((3, N), np.nan)
    stht = np.sin(tht)
    rhat[0, :] = np.cos(phi) * stht
    rhat[1, :] = np.sin(phi) * stht
    rhat[2, :] = np.cos(tht)

    integ_val, _ = runit_integrand(rhat, R, ru, vu, bs, Csp, Asinv, Asinvdot, Asinvdotdot, logAsdet, rudot, rudotdot, Q0T, Q0Tdot, Q0Tdotdot, SlowMode)

    integ = stht * integ_val

    if N == 1:
        return integ[0]
    return integ

def runit_integrand(runit, R, ru, vu, bs, Csp, Asinv, Asinvdot, Asinvdotdot, logAsdet, rudot, rudotdot, Q0T, Q0Tdot, Q0Tdotdot, SlowMode):

    sqrt2 = np.sqrt(2)
    sqrtpi = np.sqrt(np.pi)
    sqrt2pi = sqrt2 * sqrtpi

    out = {}
    out['MRstarNegatives'] = 0
    out['SRstarImaginaries'] = 0

    R2 = R**2
    twoR = 2 * R

    Nunit = runit.shape[1]

    if SlowMode:
        integ = np.full(Nunit, np.nan)
        out['QRT'] = integ.copy()
        out['QRTdot'] = integ.copy()
        out['QRTdotdot'] = integ.copy()
        out['SRstar2'] = integ.copy()

        for n in range(Nunit):
            rhat = runit[:, n]

            vup = vu + bs @ (R * rhat - ru)

            snu2 = rhat.T @ Csp @ rhat
            if snu2 <= 0:
                nusqrt2pi = max(0, -rhat.T @ vup) * sqrt2pi
            else:
                snu = np.sqrt(snu2)
                nus = (rhat.T @ vup) / snu / sqrt2
                Hnu = np.exp(-nus**2) - sqrtpi * nus * erfc(nus)
                nusqrt2pi = snu * Hnu

            Asinv_ru = Asinv @ ru
            Asinv_rhat = Asinv @ rhat

            QRT = Q0T - twoR * rhat.T @ Asinv_ru + R2 * rhat.T @ Asinv_rhat

            QRTdot = Q0Tdot - twoR * rhat.T @ (Asinvdot @ ru + Asinv @ rudot) + R2 * rhat.T @ Asinvdot @ rhat

            QRTdotdot = Q0Tdotdot - twoR * rhat.T @ (Asinvdotdot @ ru + 2 * Asinvdot @ rudot + Asinv @ rudotdot) + R2 * rhat.T @ Asinvdotdot @ rhat

            QRstar = QRT - 0.5 * QRTdot**2 / QRTdotdot

            MRstar = QRstar - logAsdet
            if MRstar < 0:
                out['MRstarNegatives'] += 1
                QRstar = logAsdet

            if QRTdotdot < 0:
                out['SRstarImaginaries'] += 1
                SRstar = np.nan
            else:
                SRstar = sqrt2 / np.sqrt(QRTdotdot)

            integ[n] = np.exp(-0.5 * QRstar) * nusqrt2pi * SRstar

            out['QRT'][n] = QRT
            out['QRTdot'][n] = QRTdot
            out['QRTdotdot'][n] = QRTdotdot
            out['SRstar2'][n] = SRstar**2

    else:
        # Vectorized
        rurep = np.tile(ru[:, np.newaxis], (1, Nunit))
        vurep = np.tile(vu[:, np.newaxis], (1, Nunit))

        vup = vurep + bs @ (R * runit - rurep)

        nusqrt2pi = np.full(Nunit, np.nan)
        snu2 = np.sum(runit * (Csp @ runit), axis=0) # (Nunit,)

        ndx = snu2 <= 0
        if np.any(ndx):
            mrdv = -np.sum(vup[:, ndx] * runit[:, ndx], axis=0)
            mrdv[mrdv < 0] = 0
            nusqrt2pi[ndx] = mrdv * sqrt2pi

        ndx = ~ndx
        if np.any(ndx):
            snu = np.sqrt(snu2[ndx])
            nus = np.sum(vup[:, ndx] * runit[:, ndx], axis=0) / snu / sqrt2
            Hnu = np.exp(-nus**2) - sqrtpi * nus * erfc(nus)
            nusqrt2pi[ndx] = snu * Hnu

        Asinv_rhat = Asinv @ runit
        out['QRT'] = np.tile(Q0T, Nunit) - R * np.sum((2 * rurep - R * runit) * Asinv_rhat, axis=0)

        rudotrep = np.tile(rudot[:, np.newaxis], (1, Nunit))
        term1 = Asinvdot @ rurep + Asinv @ rudotrep
        term2 = Asinvdot @ runit
        out['QRTdot'] = np.tile(Q0Tdot, Nunit) - twoR * np.sum(runit * term1, axis=0) + R2 * np.sum(runit * term2, axis=0)

        rudotdotrep = np.tile(rudotdot[:, np.newaxis], (1, Nunit))
        term3 = Asinvdotdot @ rurep + 2 * Asinvdot @ rudotrep + Asinv @ rudotdotrep
        term4 = Asinvdotdot @ runit
        out['QRTdotdot'] = np.tile(Q0Tdotdot, Nunit) - twoR * np.sum(runit * term3, axis=0) + R2 * np.sum(runit * term4, axis=0)

        QRstar = out['QRT'] - 0.5 * out['QRTdot']**2 / out['QRTdotdot']

        MRstar = QRstar - logAsdet
        MRstarNegatives = MRstar < 0
        out['MRstarNegatives'] = np.sum(MRstarNegatives)
        QRstar[MRstarNegatives] = logAsdet

        SRstarImaginaries = out['QRTdotdot'] <= 0
        out['SRstarImaginaries'] = np.sum(SRstarImaginaries)

        out['SRstar2'] = np.full(QRstar.shape, np.nan)
        if out['SRstarImaginaries'] > 0:
            QRTdotdotPositives = ~SRstarImaginaries
            out['SRstar2'][QRTdotdotPositives] = 2.0 / out['QRTdotdot'][QRTdotdotPositives]
        else:
            out['SRstar2'] = 2.0 / out['QRTdotdot']

        integ = np.exp(-0.5 * QRstar) * nusqrt2pi * np.sqrt(out['SRstar2'])

    return integ, out
