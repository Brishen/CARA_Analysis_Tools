import numpy as np
import warnings
from DistributedPython.ProbabilityOfCollision.Utils.jacobian_E0_to_Xt import jacobian_E0_to_Xt
from DistributedPython.Utils.OrbitTransformations.convert_cartesian_to_equinoctial import convert_cartesian_to_equinoctial
from DistributedPython.Utils.AugmentedMath.refine_bounded_extrema import refine_bounded_extrema

def PeakOverlapPos(t, xb1, Jb1, t01, Eb01, Qb01, xb2, Jb2, t02, Eb02, Qb02, HBR, params=None):
    """
    Find the inertial-frame position, rpk, of the point of peak overlap of the primary and secondary distributions
    for time = t, along with the mean primary and secondary velocities at that point.

    Args:
        t (float): Current time (s).
        xb1 (array-like): Mean inertial-frame pos/vel state for primary [6x1].
        Jb1 (array-like): Jacobian matrix dx/dE evaluated at mean state for primary [6x6].
        t01 (float): Initial time for equinoctial PDF for primary.
        Eb01 (array-like): Initial mean equinoctial element state for primary [6x1].
        Qb01 (array-like): Initial mean equinoctial covariance for primary [6x6].
        xb2 (array-like): Mean inertial-frame pos/vel state for secondary [6x1].
        Jb2 (array-like): Jacobian matrix dx/dE evaluated at mean state for secondary [6x6].
        t02 (float): Initial time for equinoctial PDF for secondary.
        Eb02 (array-like): Initial mean equinoctial element state for secondary [6x1].
        Qb02 (array-like): Initial mean equinoctial covariance for secondary [6x6].
        HBR (float): Combined primary+secondary hard-body radius.
        params (dict, optional): Structure of execution parameters.

    Returns:
        tuple: (conv, rpk, v1pk, v2pk, aux)
    """

    # Initializations and defaults
    if params is None:
        params = {}

    # Tolerances for convergence using relative Maha. distance squared
    MD2tol = params.get('MD2tol', [1e-12, 1e-6])
    if np.isscalar(MD2tol):
        MD2tol = [MD2tol, np.sqrt(MD2tol)]
    elif len(MD2tol) == 1:
        MD2tol = [MD2tol[0], np.sqrt(MD2tol[0])]

    # Maximum allowed change in relative orbital energy per iteration
    MaxRelEnergyChange = params.get('MaxRelEnergyChange', 0.10)

    # Maximum number of iterations to perform
    maxiter = params.get('maxiter', 100)

    # Parameters for iterative averaging
    avgiter = min(35, round(maxiter * 0.35))
    acciter = min(25, round(maxiter * 0.25))
    osciter = min(15, round(maxiter * 0.15))

    # Matrix to remediate SIGMAp maxtrices
    SigpRem0 = np.diag(np.tile(HBR, 3)**2)

    # Eigenvalue clipping factor
    Fclip = params.get('Fclip', 1e-4)
    Lclip = (HBR * Fclip)**2

    GM = params.get('GM', 3.986004418e5)

    verbose = params.get('verbose', False)

    # Other initializations
    twopi = 2 * np.pi
    I3x3 = np.eye(3)

    # Flatten inputs
    xb1 = np.asarray(xb1).flatten()
    Eb01 = np.asarray(Eb01).flatten()
    xb2 = np.asarray(xb2).flatten()
    Eb02 = np.asarray(Eb02).flatten()
    Jb1 = np.asarray(Jb1)
    Qb01 = np.asarray(Qb01)
    Jb2 = np.asarray(Jb2)
    Qb02 = np.asarray(Qb02)

    # Sin and Cos values for initial mean longitudes
    # Eb01[5] is lM (lambdaM)
    sinLb01 = np.sin(Eb01[5])
    sinLb02 = np.sin(Eb02[5])
    cosLb01 = np.cos(Eb01[5])
    cosLb02 = np.cos(Eb02[5])

    # Offsets of current time from the initial times of equinoctial states
    dt01 = t - t01
    dt02 = t - t02

    # Initialize estimates for expansion-center states and Jacobians
    xs1 = xb1.copy()
    Js1 = Jb1.copy()
    Es01 = Eb01.copy()

    xs2 = xb2.copy()
    Js2 = Jb2.copy()
    Es02 = Eb02.copy()

    # Initialize iteration and convergence variables
    iterating = True
    iteration = 0
    converged = False
    failure = 0
    mup_old = np.nan
    mup_old2 = np.nan
    vu1p_old = np.nan # Need to initialize these as they are used in printing
    vu2p_old = np.nan
    iterAdj = -np.inf

    aux = {}

    # Iterate to find the peak overlap position and related quantities

    while iterating:

        if verbose:
            print(f'iter = {iteration:03d}')
            if iteration == 0:
                print(f' Eb01 = {Es01}')
                print(f' Eb02 = {Es02}')

        # Difference from mean epoch eq. elements, and offset states for current iteration
        dE01 = np.zeros(6)
        dE02 = np.zeros(6)

        if iteration == 0:
            xu1 = xs1.copy()
            xu2 = xs2.copy()
        else:
            dE01[0:5] = Eb01[0:5] - Es01[0:5]
            # dE01(6) = asin(sinLb01*cos(Es01(6))-cosLb01*sin(Es01(6)));
            dE01[5] = np.arcsin(sinLb01 * np.cos(Es01[5]) - cosLb01 * np.sin(Es01[5]))

            dE02[0:5] = Eb02[0:5] - Es02[0:5]
            dE02[5] = np.arcsin(sinLb02 * np.cos(Es02[5]) - cosLb02 * np.sin(Es02[5]))

            xu1 = xs1 + Js1 @ dE01
            xu2 = xs2 + Js2 @ dE02

        # Extract pos & vel vectors from offset states
        ru1 = xu1[0:3]
        vu1 = xu1[3:6]
        ru2 = xu2[0:3]
        vu2 = xu2[3:6]

        # Calculate pos/vel covariances, enforcing symmetry
        Ps1 = Js1 @ Qb01 @ Js1.T
        Ps1 = (Ps1 + Ps1.T) / 2
        aux['Ps1'] = Ps1

        Ps2 = Js2 @ Qb02 @ Js2.T
        Ps2 = (Ps2 + Ps2.T) / 2
        aux['Ps2'] = Ps2

        # Decompose pos/vel covariances into 3x3 submatrices
        As1 = Ps1[0:3, 0:3]
        Bs1 = Ps1[3:6, 0:3]

        As2 = Ps2[0:3, 0:3]
        Bs2 = Ps2[3:6, 0:3]

        # Calculate inverses of position covariances using eigenvalue clipping
        Lraw1, Veig1 = np.linalg.eigh(As1)
        Leig1 = Lraw1.copy()
        Leig1[Leig1 < Lclip] = Lclip
        As1inv = (Veig1 * (1.0 / Leig1)) @ Veig1.T

        Lraw2, Veig2 = np.linalg.eigh(As2)
        Leig2 = Lraw2.copy()
        Leig2[Leig2 < Lclip] = Lclip
        As2inv = (Veig2 * (1.0 / Leig2)) @ Veig2.T

        # Calculate the covariance of the peak overlap position overlap distribution
        Sigpinv = As1inv + As2inv
        LrawSig, VeigSig = np.linalg.eigh(Sigpinv)
        LeigSig = LrawSig.copy()
        LeigSig[LeigSig < Lclip] = Lclip
        Sigp = (VeigSig * (1.0 / LeigSig)) @ VeigSig.T

        # Calculate the peak overlap position
        mup = Sigp @ (As1inv @ ru1 + As2inv @ ru2)

        # Remediate convergence by convolving SIGMAp matrices with sphere of radius HBR
        SigpRem = Sigp + SigpRem0
        SigpReminv = np.linalg.solve(SigpRem, I3x3)

        # Iterative processing

        if maxiter <= 1:

            # Iterations forcibly discontinued after one iteration.
            iterating = False
            converged = True

            # C12a estimates for vu1-prime and vu2-prime
            vu1p = vu1.copy()
            vu2p = vu2.copy()

        else:

            # Do mu-point averaging for iterations past max limit to accelerate slow convergence cases
            if iteration > avgiter and iteration > iterAdj + 2:
                mup = 0.5 * (mup + mup_old)

            # Calculate new center-of-linearization(CoL) points
            beta1 = Bs1 @ As1inv
            rs1, Energy1, Adj1 = AdjustPOPCoL(mup, xs1[0:3], beta1, xu1, GM, MaxRelEnergyChange, verbose)

            beta2 = Bs2 @ As2inv
            rs2, Energy2, Adj2 = AdjustPOPCoL(mup, xs2[0:3], beta2, xu2, GM, MaxRelEnergyChange, verbose)

            # Record iteration of last adjustment
            if Adj1 != 1 or Adj2 != 1:
                iterAdj = iteration

            # New estimates for vu1-prime and vu2-prime
            vu1p = vu1 + beta1 @ (rs1 - ru1)
            vu2p = vu2 + beta2 @ (rs2 - ru2)

            # Report quantities for testing
            if verbose:
                print(f' vu1p = {vu1p}')
                print(f' vu2p = {vu2p}')
                d1 = xs1[0:3] - rs1
                Msq1 = d1.T @ As1inv @ d1
                d2 = xs2[0:3] - rs2
                Msq2 = d2.T @ As2inv @ d2
                print(f' Msq1,2 = {Msq1} {Msq2}')
                print(f' Energy1,2 = {Energy1} {Energy2}')
                print(f' Adj1,2 = {Adj1} {Adj2}')

            # Mark as unconverged if an unbound orbit was encountered
            if (Energy1 >= 0) or (Energy2 >= 0):
                iterating = False
                converged = False
                # failure = 10*(Energy1 >= 0)+(Energy2 >= 0); % 11, 10, or 01
                failure = 10 * int(Energy1 >= 0) + int(Energy2 >= 0)

            else:

                # New estimates for expansion-center cartesian states
                xs1 = np.concatenate((rs1, vu1p))
                xs2 = np.concatenate((rs2, vu2p))

                # Iteration and convergence processing
                if iteration > 0:

                    # Check for convergence
                    dmup = mup_old - mup
                    dMD2 = dmup.T @ SigpReminv @ dmup

                    if dMD2 <= MD2tol[0] and iteration != iterAdj:
                        iterating = False
                        converged = True
                    elif iteration >= maxiter:
                        iterating = False
                        converged = False
                    elif iteration > osciter and iteration > iterAdj + 2:
                        # Check for back and forth oscillating convergence
                        dmuposc = mup_old2 - mup
                        dMD2osc = dmuposc.T @ SigpReminv @ dmuposc

                        if dMD2osc <= MD2tol[0]:
                            iterating = False
                            converged = 0.5
                        else:
                            if iteration > avgiter:
                                MD2cut = MD2tol[1]
                                omfrc = 0
                            elif iteration <= acciter:
                                MD2cut = MD2tol[0]
                                omfrc = 1
                            else:
                                frc = (iteration - acciter) / (avgiter - acciter)
                                omfrc = 1 - frc
                                # MD2cut = exp( omfrc*log(params.MD2tol(1)) + frc  *log(params.MD2tol(2)) );
                                MD2cut = np.exp(omfrc * np.log(MD2tol[0]) + frc * np.log(MD2tol[1]))

                            if dMD2 <= MD2cut:
                                iterating = False
                                converged = 0.1 + omfrc / 4.0
                    else:
                        dMD2osc = np.nan

                    if verbose:
                         print(f' rsdiff = {np.linalg.norm(mup_old-mup)} '
                               f' vs1dif = {np.linalg.norm(vu1p_old-vu1p) if "vu1p_old" in locals() else "NaN"} '
                               f' vs2dif = {np.linalg.norm(vu2p_old-vu2p) if "vu2p_old" in locals() else "NaN"} '
                               f' dMD2 = {dMD2} '
                               f' dMD2osc = {dMD2osc}')

                # Calculate equinoctial elements
                # Check for undefined
                # convert_cartesian_to_equinoctial returns None on failure if we handle it that way,
                # but the python function returns Nones.

                # We need to suppress warnings potentially if verbose is low, but the function takes issue_warnings
                # PeakOverlapPos.m passes 'verbose' to convert_cartesian_to_equinoctial
                # [a1s,n1s,af1s,ag1s,chi1s,psi1s,lM1s] = convert_cartesian_to_equinoctial(xs1(1:3),xs1(4:6),[],[],verbose);

                a1s, n1s, af1s, ag1s, chi1s, psi1s, lM1s, _ = convert_cartesian_to_equinoctial(
                    xs1[0:3], xs1[3:6], issue_warnings=verbose
                )

                a2s, n2s, af2s, ag2s, chi2s, psi2s, lM2s, _ = convert_cartesian_to_equinoctial(
                    xs2[0:3], xs2[3:6], issue_warnings=verbose
                )

                bad1s = (n1s is None) or np.isnan(a1s)
                bad2s = (n2s is None) or np.isnan(a2s)

                if bad1s or bad2s:
                    iterating = False
                    converged = False
                    failure = 1000 * int(bad1s) + 100 * int(bad2s)

                else:
                    # Check for unbound orbits
                    esq1s = af1s**2 + ag1s**2
                    unbound1s = (a1s <= 0) or (esq1s >= 1)

                    esq2s = af2s**2 + ag2s**2
                    unbound2s = (a2s <= 0) or (esq2s >= 1)

                    if unbound1s or unbound2s:
                        iterating = False
                        converged = False
                        failure = 10 * int(unbound1s) + int(unbound2s)

                    else:
                        # Epoch mean longitudes of the POP states at the initial times
                        lM10s = (lM1s - n1s * dt01) % twopi
                        lM20s = (lM2s - n2s * dt02) % twopi

                        # Epoch equinoctial states
                        Es01 = np.array([n1s, af1s, ag1s, chi1s, psi1s, lM10s])
                        Es02 = np.array([n2s, af2s, ag2s, chi2s, psi2s, lM20s])

                        if verbose:
                            print(f' Es01 = {Es01}')
                            print(f' Es02 = {Es02}')

                        # Epoch equinoctial Jacobians for the POP states
                        Js1, _ = jacobian_E0_to_Xt(dt01, Es01)
                        # jacobian_E0_to_Xt returns (NT, 6, 6) or (6, 6) depending on implementation.
                        # My implementation returns (NT, 6, 6).
                        # Since dt01 is scalar, it returns (1, 6, 6).
                        if Js1.ndim == 3:
                            Js1 = Js1[0]

                        Js2, _ = jacobian_E0_to_Xt(dt02, Es02)
                        if Js2.ndim == 3:
                            Js2 = Js2[0]

                        # Save results from this iteration
                        if iterating:
                            mup_old2 = mup_old
                            mup_old = mup.copy()
                            vu1p_old = vu1p.copy()
                            vu2p_old = vu2p.copy()
                            iteration += 1

    # Assemble the output quantities
    conv = converged
    rpk = mup
    v1pk = vu1p
    v2pk = vu2p

    aux['converged'] = converged
    aux['iteration'] = iteration
    aux['failure'] = failure
    aux['iterAdj'] = iterAdj

    if converged:
        aux['xs1'] = xs1
        aux['Js1'] = Js1
        aux['Es01'] = Es01

        aux['xs2'] = xs2
        aux['Js2'] = Js2
        aux['Es02'] = Es02

        aux['Sigp'] = Sigp
        aux['Sigpinv'] = Sigpinv
        aux['SigpRem'] = SigpRem
        aux['SigpReminv'] = SigpReminv

        if maxiter <= 1:
            aux['xu1'] = xb1
            aux['dE01'] = np.zeros(6)
            aux['xu2'] = xb2
            aux['dE02'] = np.zeros(6)
        else:
            dE01[0:5] = Eb01[0:5] - Es01[0:5]
            dE01[5] = np.arcsin(sinLb01 * np.cos(Es01[5]) - cosLb01 * np.sin(Es01[5]))

            dE02[0:5] = Eb02[0:5] - Es02[0:5]
            dE02[5] = np.arcsin(sinLb02 * np.cos(Es02[5]) - cosLb02 * np.sin(Es02[5]))

            aux['xu1'] = xs1 + Js1 @ dE01
            aux['dE01'] = dE01

            aux['xu2'] = xs2 + Js2 @ dE02
            aux['dE02'] = dE02

    if verbose:
        print(f'Converged = {converged} at t = {t}')

    return conv, rpk, v1pk, v2pk, aux

# Helper functions

def AdjustPOPCoL(mup, rs, beta, Xu, GM, MaxRelEnergyChange, verbose):
    """
    Adjust peak overlap position (POP) center-of-linearization (CoL) point
    to allow only a maximum relative energy change.
    """

    # Parameters for refine_bisection_search (RBE)
    RBE_Ninitial = 201
    RBE_Ninitcut = max(5, int(round(0.05 * RBE_Ninitial)))
    RBE_Nbisectmax = 100
    RBE_TolX = [5e-4, np.nan]
    RBE_TolY = [np.nan, np.nan]
    RBE_check_inputs = False

    # Energy at original CoL estimate, A = rs, and at the nominal POP CoL estimate, B = mup
    EnergyAB, _ = CalcCoLEnergy([0.0, 1.0], rs, mup, Xu, beta, GM)
    EnergyA = EnergyAB[0]
    EnergyB = EnergyAB[1]

    # Return with no adjustment if EnergyA is not negative
    if EnergyA >= 0:
        if verbose and not np.isinf(MaxRelEnergyChange):
            warnings.warn('Inputs have invalid orbital energies')
        rsAdj = mup
        EnAdj = EnergyB
        fracAdj = 1.0
        return rsAdj, EnAdj, fracAdj

    # Relative orbital energy change
    FracEnergyChange = abs(EnergyB - EnergyA) / abs(EnergyA)

    # Return nominal CoL estimate if relative energy change is acceptable
    if FracEnergyChange <= MaxRelEnergyChange:
        rsAdj = mup
        EnAdj = EnergyB
        fracAdj = 1.0
        return rsAdj, EnAdj, fracAdj

    # Anonymous function for refine_bounded_extrema (RBE) function
    fun = lambda xx: (np.abs(CalcCoLEnergy(xx, rs, mup, Xu, beta, GM)[0] / EnergyA - 1.0) - MaxRelEnergyChange)**2

    # Initial points for RBE bisection search
    xinit = np.linspace(0, 1, RBE_Ninitial)
    yinit = fun(xinit)

    # Perform RBE bisection search only if min point is too close to original CoL
    imin = np.argmin(yinit)

    if imin < RBE_Ninitcut:
        # Include endpoints in RBE if very close to edge
        # MATLAB: if imin <= 2 (indices 1, 2)
        # Python: imin <= 1 (indices 0, 1)
        endpoints = imin <= 1

        xmnma, ymnma, _, _, converged, _, _, _, _, _ = refine_bounded_extrema(
            fun, xinit, yinit, Ninitial=None, Nbisectmax=RBE_Nbisectmax, extrema_types=1,
            TolX=RBE_TolX, TolY=RBE_TolY, endpoints=endpoints, verbose=int(verbose > 1), check_inputs=RBE_check_inputs
        )

        if verbose and not converged:
            warnings.warn('Bisection search did not converge')

        Nmnma = xmnma.size
        if Nmnma == 0:
            if verbose:
                warnings.warn('Bisection search did not find any minima')
            xmin = xinit[imin] # Fallback to grid min
        else:
            imin_refined = np.argmin(ymnma)
            xmin = xmnma[imin_refined]
    else:
        xmin = xinit[imin]

    # Ensure adjusted point is not original CoL itself
    if xmin <= 0:
        xmin = xinit[1] # Use second point

    # Define output quantities
    fracAdj = xmin
    Energy, rsAdj = CalcCoLEnergy([fracAdj], rs, mup, Xu, beta, GM)
    EnAdj = Energy[0]
    rsAdj = rsAdj[:, 0]

    return rsAdj, EnAdj, fracAdj

def CalcCoLEnergy(x, rsA, rsB, Xu, beta, GM):
    """
    Vectorized center-of-linearization (CoL) orbit energy calculation
    """
    x = np.asarray(x).flatten()
    N = x.size

    rsA = np.asarray(rsA).flatten()
    rsB = np.asarray(rsB).flatten()
    Xu = np.asarray(Xu).flatten()

    # Vectorized calculation of intermediate positions: rs = rsA + x*(rsB-rsA)
    # rs shape: (3, N)

    diff_rs = (rsB - rsA).reshape(3, 1)
    rsA_col = rsA.reshape(3, 1)

    rs = rsA_col + diff_rs * x.reshape(1, N) # (3, 1) + (3, 1) * (1, N) = (3, N)

    # Magnitudes of intermediate positions
    rsmag = np.sqrt(np.sum(rs**2, axis=0)) # (N,)

    # Vectorized calculation of conditional velocities:
    # vup = vu + beta*(rs-ru)    with    beta = Bs*Asinv

    ru = Xu[0:3].reshape(3, 1)
    vu = Xu[3:6].reshape(3, 1)

    # vup = vu + beta @ (rs - ru)
    # rs - ru shape (3, N)

    vup = vu + beta @ (rs - ru) # (3, 1) + (3, 3) @ (3, N) = (3, N)

    # Magnitude of squared conditional velocity
    vup2 = np.sum(vup**2, axis=0) # (N,)

    # Vector of CoL energies calculated using conditional velocities (N,)
    Energy = vup2 / 2.0 - GM / rsmag

    # Return Energy as (N,) array and rs as (3, N)
    return Energy, rs
