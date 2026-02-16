import numpy as np
from DistributedPython.Utils.OrbitTransformations.convert_equinoctial_to_cartesian import convert_equinoctial_to_cartesian

def delta_r2_equin(Ts,
    nmean10, afmean10, agmean10, chimean10, psimean10, lMmean10,
    nmean20, afmean20, agmean20, chimean20, psimean20, lMmean20,
    GM):
    """
    Calculate the distance squared between two objects in two-body orbits.

    Parameters
    ----------
    Ts : float or array_like
        Time offsets from epoch (s).
    nmean10, afmean10, agmean10, chimean10, psimean10, lMmean10 : float
        Equinoctial elements for object 1.
    nmean20, afmean20, agmean20, chimean20, psimean20, lMmean20 : float
        Equinoctial elements for object 2.
    GM : float
        Gravitational constant.

    Returns
    -------
    dr2 : float or ndarray
        Squared distance between the two objects.
    """

    # convert_equinoctial_to_cartesian returns a tuple, the first element is rvec (3, N)
    r1s = convert_equinoctial_to_cartesian(
        nmean10, afmean10, agmean10, chimean10, psimean10, lMmean10,
        Ts, 1, GM)[0]

    r2s = convert_equinoctial_to_cartesian(
        nmean20, afmean20, agmean20, chimean20, psimean20, lMmean20,
        Ts, 1, GM)[0]

    drs = r2s - r1s
    dr2 = np.sum(drs * drs, axis=0)

    return dr2
