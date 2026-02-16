import numpy as np
from DistributedPython.ProbabilityOfCollision.Utils.jacobian_E0_to_Xt import jacobian_E0_to_Xt
from DistributedPython.ProbabilityOfCollision.Utils.PeakOverlapPos import PeakOverlapPos

def PeakOverlapMD2(t, t10, Eb10, Qb10, t20, Eb20, Qb20, HBR, EMD2, params=None):
    """
    Calculate the effective Maha. distance using the peak overlap point.

    Actually, calculate MD^2 + log(det(As)), to account for slight variations
    in the determinant of the As matrix, which shows up in the denominator.

    Args:
        t (float): Current time (s).
        t10 (float): Initial time for equinoctial PDF for primary.
        Eb10 (array-like): Initial mean equinoctial element state for primary [6x1].
        Qb10 (array-like): Initial mean equinoctial covariance for primary [6x6].
        t20 (float): Initial time for equinoctial PDF for secondary.
        Eb20 (array-like): Initial mean equinoctial element state for secondary [6x1].
        Qb20 (array-like): Initial mean equinoctial covariance for secondary [6x6].
        HBR (float): Combined primary+secondary hard-body radius.
        EMD2 (int): Mode (1 or 2).
        params (dict, optional): Structure of execution parameters.

    Returns:
        tuple: (MD2, Xu, Ps, Asdet, Asinv, POPconverged, aux)
    """

    if params is None:
        params = {}

    # Calc pos/vel mean states and associated Jacobians at specified time
    Jb1, xb1 = jacobian_E0_to_Xt(t, Eb10)
    if Jb1.ndim == 3: Jb1 = Jb1[0]
    if xb1.ndim == 2: xb1 = xb1.flatten()

    Jb2, xb2 = jacobian_E0_to_Xt(t, Eb20)
    if Jb2.ndim == 3: Jb2 = Jb2[0]
    if xb2.ndim == 2: xb2 = xb2.flatten()

    # Calc peak overlap position
    POPconverged, _, _, _, POP = PeakOverlapPos(
        t, xb1, Jb1, t10, Eb10, Qb10, xb2, Jb2, t20, Eb20, Qb20, HBR, params
    )

    aux = {}

    if POPconverged:

        # Relative POP-corrected distance
        Xu = POP['xu2'] - POP['xu1']
        ru = Xu[0:3]

        # Construct the POP-corrected joint pos/vel covariance
        XCprocessing = params.get('XCprocessing', False)

        if XCprocessing:
            # DCP 6x1 sensitivity vectors for cartesian pos/vel
            # state at the current eph time

            # Need GEp and GEs and sigpXsigs from params
            GEp = params.get('GEp')
            GEs = params.get('GEs')
            sigpXsigs = params.get('sigpXsigs')

            # POP['Js1'] should be available
            # Ensure GEp/GEs are treated as column vectors for multiplication
            # POP['Js1'] (6,6) @ GEp (6,) -> GCp (6,)
            GCp = POP['Js1'] @ GEp
            GCs = POP['Js2'] @ GEs

            # Calculate outer products for sensitivity correction
            # MATLAB: GCs*GCp' + GCp*GCs'
            term = sigpXsigs * (np.outer(GCs, GCp) + np.outer(GCp, GCs))

            Ps = POP['Ps1'] + POP['Ps2'] - term
        else:
            Ps = POP['Ps1'] + POP['Ps2']

        # Calculate the inverse of A, remediating NPDs with eigenvalue clipping
        As = Ps[0:3, 0:3]
        Fclip = params.get('Fclip', 1e-4)
        Lclip = (HBR * Fclip)**2

        Lraw, Veig = np.linalg.eigh(As)
        Leig = Lraw.copy()
        Leig[Leig < Lclip] = Lclip

        Asdet = np.prod(Leig)
        Asinv = (Veig * (1.0 / Leig)) @ Veig.T

        aux['As'] = As
        aux['AsLeig'] = Leig
        aux['AsVeig'] = Veig
        aux['AsLclip'] = Lclip

        # Calculate effective Maha distance
        MD2 = ru.T @ Asinv @ ru
        aux['MD2actual'] = MD2

        # Calculate the "effective" or modified Maha distance
        if EMD2 == 1:
            MD2 = MD2 + np.log(Asdet)
        elif EMD2 == 2:
            MD2 = MD2 + np.log(Asdet) - 2 * np.log(np.linalg.norm(Xu[3:6]))

    else:
        # Return null values for no POP convergence
        MD2 = np.nan
        Xu = None
        Ps = None
        Asdet = None
        Asinv = None
        aux = {}

    return MD2, Xu, Ps, Asdet, Asinv, POPconverged, aux
