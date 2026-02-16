import warnings
from DistributedPython.ProbabilityOfCollision.Utils.set_default_param import set_default_param
from DistributedPython.ProbabilityOfCollision.PcMultiStepWithSDMC import PcMultiStepWithSDMC

def PcMultiStepWithPlots(r1, v1, C1, r2, v2, C2, HBR, params=None):
    """
    Executes the function PcMultiStepWithSDMC and makes temporal and close approach
    distribution plots (Not Implemented).

    Parameters
    ----------
    r1, v1, r2, v2 : array_like
        State vectors.
    C1, C2 : array_like
        Covariance matrices.
    HBR : float
        Hard-body radius.
    params : dict, optional
        Parameter dictionary.

    Returns
    -------
    Pc : float
        Recommended Pc value.
    out : dict
        Auxiliary output information.
    """
    if params is None:
        params = {}

    params = set_default_param(params, 'generate_conj_plots', [True, True])

    # Call PcMultiStepWithSDMC
    Pc, out = PcMultiStepWithSDMC(r1, v1, C1, r2, v2, C2, HBR, params)

    if params['generate_conj_plots'][0] or params['generate_conj_plots'][1]:
        warnings.warn("Plotting functionality in PcMultiStepWithPlots is not implemented in this Python port.")
        out['pcTimePlotFile'] = ''
        out['caDistPlotFile'] = ''

    return Pc, out
