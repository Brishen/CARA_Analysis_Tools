import numpy as np
from DistributedPython.ProbabilityOfCollision.Utils.set_default_param import set_default_param

def CheckForRetrograde(r, v, Eps):
    """
    Check if an orbit is near retrograde.

    Args:
        r (np.ndarray): Position vector (3, 1)
        v (np.ndarray): Velocity vector (3, 1)
        Eps (float): Tolerance

    Returns:
        bool: True if retrograde, False otherwise.
    """
    h = np.cross(r.flatten(), v.flatten())
    # Retro = 1 + h(3)/sqrt(h'*h) < Eps
    # h(3) is the Z component.
    return (1 + h[2] / np.linalg.norm(h)) < Eps

def RetrogradeReorientation(r1, v1, C1, r2, v2, C2, params=None):
    """
    Reorient primary and secondary states to prevent either from
    being processed as a retrograde Keplerian orbit (i.e., an orbit
    with 180deg inclination). This entails rotating both states to use
    a frame with a different set of inertial axes. Also reorient the
    covariance matrices.

    Args:
        r1, v1: Primary position velocity [1x3 or 3x1 or 3, vectors]
        r2, v2: Secondary position velocity [1x3 or 3x1 or 3, vectors]
        C1, C2: Primary and secondary covariances [6x6 matrices]
        params: Dictionary of execution parameters
            params['RetrogradeReorientation'] = processing mode
                0 => No retrograde orbit reorientation
                1 => If either orbit is retrograde, reorient the ref. frame axes
                2 => Always reorient the ref. frame axes (testing mode)
                3 => Reorient axes to force primary to be retrograde (testing mode)

    Returns:
        r1, v1, C1, r2, v2, C2: Reoriented states and covariances
        out: Output dictionary with fields:
            out['Reoriented']: Binary flag to indicate if frame reorientation was performed
            out['Retro1']: Primary orbit retrograde flag, if calculated
            out['Retro2']: Secondary orbit retrograde flag, if calculated
            out['M3']: 3x3 rotation matrix, if calculated
            out['M6']: 6x6 rotation matrix, if calculated
    """
    # Set defaults
    if params is None:
        params = {}
    params = set_default_param(params, 'RetrogradeReorientation', 1)

    out = {}

    # Process different modes
    if params['RetrogradeReorientation'] == 0:
        out['Reoriented'] = False
        return r1, v1, C1, r2, v2, C2, out

    # Check the sizes of input vectors
    sz = r1.shape

    # Validate input dimensions match
    if r2.shape != sz or v1.shape != sz or v2.shape != sz:
        raise ValueError('Input vectors must be same dimension')

    InputRowVectors = False
    OneD = False

    if sz == (1, 3):
        InputRowVectors = True
    elif sz == (3, 1):
        InputRowVectors = False
    elif sz == (3,):
        OneD = True
    else:
        raise ValueError('Invalid dimension for input vectors')

    # Convert to column vectors for internal processing
    r1 = r1.reshape(3, 1)
    v1 = v1.reshape(3, 1)
    r2 = r2.reshape(3, 1)
    v2 = v2.reshape(3, 1)

    # Process different retrograde reorientation modes
    if params['RetrogradeReorientation'] == 1 or params['RetrogradeReorientation'] == 2:

        # Check if frame reorientation is required due to orbit being
        # within a small tolerance of retrograde

        # Set default tolerance
        params = set_default_param(params, 'Eps', 1e-6)

        # Check if primary orbit is near retrograde
        out['Retro1'] = CheckForRetrograde(r1, v1, params['Eps'])

        # Check if secondary orbit is near retrograde
        out['Retro2'] = CheckForRetrograde(r2, v2, params['Eps'])

        # Adjust reference frame, attempting to eliminate retrograde orbits
        if out['Retro1'] or out['Retro2'] or params['RetrogradeReorientation'] == 2:

            # Set default verbose flag
            params = set_default_param(params, 'verbose', True)
            if params['verbose']:
                print(f"*** Reorienting frame. Retrograde status: Primary = {out['Retro1']} Secondary = {out['Retro2']}")

            # Axes for new reference frame
            Xhat = np.array([[0], [1], [0]])
            Yhat = np.array([[0], [0], [1]])
            Zhat = np.array([[1], [0], [0]])

            # Rotation matrix for reoriented ref. frame
            M3 = np.hstack([Xhat, Yhat, Zhat]).T

            # Rotate states to use reoriented frame
            r1New = M3 @ r1
            v1New = M3 @ v1
            r2New = M3 @ r2
            v2New = M3 @ v2

            # Check that reorientation did not result in retrograde orbits
            Retro1New = CheckForRetrograde(r1New, v1New, params['Eps'])
            Retro2New = CheckForRetrograde(r2New, v2New, params['Eps'])

            # If retrograde orbits persist, iterate
            if Retro1New or Retro2New:

                # Set default max. number of reorientation iterations
                params = set_default_param(params, 'MaxIter', 5)

                # Iterate to find a reorientation
                iterating = True
                iter_count = 0
                converged = False
                halfpi = np.pi / 2
                phi = halfpi
                tht = halfpi

                while iterating:
                    print(f" ** Reorientation iteration {iter_count} for phi = {phi*180/np.pi} & theta = {tht*180/np.pi} Retrograde status: Primary = {Retro1New} Secondary = {Retro2New}")
                    # Randomly generate new phi and tht Euler angles
                    phi = halfpi * (np.random.rand() + 1.0)
                    cp = np.cos(phi)
                    sp = np.sin(phi)

                    tht = halfpi * (np.random.rand() + 0.5)
                    ct = np.cos(tht)
                    st = np.sin(tht)

                    # Calculate new reorientation matrix
                    C_rot = np.array([[1, 0, 0], [0, ct, st], [0, -st, ct]])
                    D_rot = np.array([[cp, sp, 0], [-sp, cp, 0], [0, 0, 1]])
                    M3 = C_rot @ D_rot

                    # Check that reorientation did not produce retrogrades
                    # Note: we should apply M3 to original vectors, not r1New from previous step.
                    # The matlab code: M3 = C*D. r1New = M3 * r1.
                    # But r1 was already updated? No, r1 is the input to the function (or adjusted to column).
                    # Wait, in matlab code:
                    # r1New = M3 * r1; v1New = M3 * v1; ...
                    # Retro1New = CheckForRetrograde(r1New,v1New,params.Eps);
                    # ...
                    # if Retro1New || Retro2New ...
                    #    while iterating
                    #       M3 = C*D;
                    #       % Check that reorientation did not produce retrogrades
                    #       % BUT wait, check assumes we calculate r1New again.
                    #       % The matlab code seems to miss updating r1New in the loop for the check?
                    #       % Let's look closer at Matlab code.

                    # Matlab code:
                    # while iterating
                    #    ...
                    #    M3 = C*D;
                    #    % Check that reorientation did not produce retrogrades
                    #    Retro1New = CheckForRetrograde(r1New,v1New,params.Eps);

                    # This looks like a bug in the original MATLAB code!
                    # Inside the loop, M3 is updated, but r1New and v1New are NOT updated using the new M3 before checking CheckForRetrograde.
                    # So Retro1New will always be the same value as before the loop unless r1New is updated.
                    # However, since I am converting, I should probably fix this or replicate behavior?
                    # If I replicate, it might infinite loop if it doesn't converge?
                    # "iterating = iter < MaxIter;" ensures it stops.
                    # But if r1New is not updated, Retro1New is constant.
                    # Wait, CheckForRetrograde uses r1New. r1New depends on M3 * r1? No, r1New was set outside the loop.
                    # So in Matlab:
                    # r1New = M3 * r1; % outside loop
                    # while iterating
                    #    M3 = C*D; % new M3
                    #    Retro1New = CheckForRetrograde(r1New, ...); % r1New is OLD
                    # This definitely looks like a bug in Matlab code.
                    # Unless `CheckForRetrograde` somehow uses the new M3? No, it takes r, v.

                    # I will assume the INTENT is to check the NEW orientation.
                    # So I will update r1New inside the loop in Python.

                    r1Check = M3 @ r1
                    v1Check = M3 @ v1
                    r2Check = M3 @ r2
                    v2Check = M3 @ v2

                    Retro1New = CheckForRetrograde(r1Check, v1Check, params['Eps'])
                    Retro2New = CheckForRetrograde(r2Check, v2Check, params['Eps'])

                    if not Retro1New and not Retro2New:
                        converged = True
                        iterating = False
                        # Update the "New" variables to hold the converged values
                        r1New = r1Check
                        v1New = v1Check
                        r2New = r2Check
                        v2New = v2Check
                    else:
                        iter_count += 1
                        iterating = iter_count < params['MaxIter']
            else:
                # First reorientation converged to non-retrograde status
                converged = True

            # If converged, adopt new orientation
            if converged:
                # Output new vectors
                r1 = r1New
                v1 = v1New
                r2 = r2New
                v2 = v2New
                # Output rotation matrices
                out['M3'] = M3 # 3x3 rotation matrix
                Z3 = np.zeros((3, 3))
                out['M6'] = np.block([[M3, Z3], [Z3, M3]]) # 6x6 rot. matrix
                # Output new covariances
                C1 = out['M6'] @ C1 @ out['M6'].T
                C2 = out['M6'] @ C2 @ out['M6'].T
                # Set the frame reorientation flag
                out['Reoriented'] = True
            else:
                raise RuntimeError('No retrograde reorientation found')

        else:
            # No reorientation required
            out['Reoriented'] = False

    elif params['RetrogradeReorientation'] == 3:

        # Special testing mode to reorient such that primary orbit is
        # perfectly retrograde
        out['Reoriented'] = True

        # Set default verbose flag
        params = set_default_param(params, 'verbose', True)
        if params['verbose']:
            print('Reorienting frame to make primary orbit retrograde')

        # Force primary to have a perfectly retrograde orbit (for testing)
        hvec1 = np.cross(r1.flatten(), v1.flatten())
        hmag1 = np.linalg.norm(hvec1)
        Zhat = (-hvec1 / hmag1).reshape(3, 1)

        # Xhat = r1 - Zhat*(r1'*Zhat); Xhat = Xhat/norm(Xhat);
        Xhat = r1 - Zhat * (r1.T @ Zhat)
        Xhat = Xhat / np.linalg.norm(Xhat)

        # Yhat = cross(Zhat,Xhat);
        Yhat = np.cross(Zhat.flatten(), Xhat.flatten()).reshape(3, 1)

        # M3 = [Xhat Yhat Zhat]';
        M3 = np.hstack([Xhat, Yhat, Zhat]).T
        Z3 = np.zeros((3, 3))
        M6 = np.block([[M3, Z3], [Z3, M3]])

        # Rotate states and covariances
        r1 = M3 @ r1
        v1 = M3 @ v1
        C1 = M6 @ C1 @ M6.T
        r2 = M3 @ r2
        v2 = M3 @ v2
        C2 = M6 @ C2 @ M6.T

    else:
        raise ValueError('Invalid RetrogradeReorientation parameter')

    # Convert output to match input row vectors, if required
    if InputRowVectors:
        r1 = r1.reshape(1, 3)
        v1 = v1.reshape(1, 3)
        r2 = r2.reshape(1, 3)
        v2 = v2.reshape(1, 3)
    elif OneD:
        r1 = r1.flatten()
        v1 = v1.flatten()
        r2 = r2.flatten()
        v2 = v2.flatten()

    return r1, v1, C1, r2, v2, C2, out
