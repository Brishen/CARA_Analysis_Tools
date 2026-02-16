import numpy as np

def FindNearbyCA(X1, X2, MotionMode='LINEAR', RelTol=1e-12):
    """
    FindNearbyCA - Find the close-approach (CA) point from an input primary
                   and secondary position/velocity inertial state.

    Args:
        X1 (np.ndarray): Primary object's pos/vel state vector in ECI coordinates (6x1) or (6,)
        X2 (np.ndarray): Secondary object's pos/vel state vector in ECI coordinates (6x1) or (6,)
        MotionMode (str, optional): 'LINEAR' (currently implemented) or 'TWOBODY'. Defaults to 'LINEAR'.
        RelTol (float, optional): Tolerance for finding CA. Defaults to 1e-12.

    Returns:
        tuple:
            dTCA (float): Offset time to CA (s)
            X1CA (np.ndarray): Primary ECI state at CA (6x1)
            X2CA (np.ndarray): Secondary ECI state at CA (6x1)
    """
    # Set up defaults
    if MotionMode is None:
        MotionMode = 'LINEAR'
    else:
        MotionMode = MotionMode.upper()

    X1 = np.asarray(X1).flatten()
    X2 = np.asarray(X2).flatten()

    if X1.size != 6 or X2.size != 6:
        raise ValueError('X1 and X2 must be 6-element vectors')

    if MotionMode == 'LINEAR':
        # Linear motion mode
        motion = 1
    elif MotionMode == 'TWOBODY':
        # 2-body motion mode
        motion = 2
    else:
        raise ValueError('Invalid motion mode')

    if motion == 1: # Linear motion
        # Primary and secondary cartesian positions and velocities
        r1 = X1[0:3]
        v1 = X1[3:6]
        r2 = X2[0:3]
        v2 = X2[3:6]

        # Relative velocity
        v = v2 - v1
        vmag2 = np.dot(v, v)

        if vmag2 == 0:
            # No TCA offset can be calculated for zero relative velocity
            dTCA = np.nan
            r1CA = r1
            r2CA = r2
        else:
            # TCA offset for linear relative motion
            # dTCA = -((r2-r1)'*v)/vmag2;
            dTCA = -np.dot(r2 - r1, v) / vmag2

            # Primary and secondary positions at the linear-motion TCA
            r1CA = r1 + dTCA * v1
            r2CA = r2 + dTCA * v2

        # Return final linear motion states
        X1CA = np.concatenate((r1CA, v1))
        X2CA = np.concatenate((r2CA, v2))

    elif motion == 2: # Two-body motion
        raise NotImplementedError('TWOBODY motion mode not implemented')

    return dTCA, X1CA, X2CA
