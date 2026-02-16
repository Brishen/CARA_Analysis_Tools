import numpy as np
import warnings
from DistributedPython.Utils.OrbitTransformations.convert_cartesian_to_equinoctial import convert_cartesian_to_equinoctial
from DistributedPython.Utils.OrbitTransformations.jacobian_equinoctial_to_cartesian import jacobian_equinoctial_to_cartesian
from DistributedPython.ProbabilityOfCollision.Utils.CovRemEigValClip import CovRemEigValClip

def EquinoctialMatrices(r, v, C, RemEqCov):
    """
    Calculate equinoctial state, covariance and Jacobian matrices

    Args:
        r (array-like): Position vector (m).
        v (array-like): Velocity vector (m/s).
        C (array-like): Position/Velocity covariance matrix (m, m/s).
        RemEqCov (bool): Remediate equinoctial covariance if required.

    Returns:
        tuple:
            X (np.ndarray): Cartesian state vector (km, km/s).
            P (np.ndarray): Cartesian covariance matrix (km, km/s).
            E (np.ndarray): Equinoctial state vector.
            J (np.ndarray): Jacobian matrix dX/dE.
            K (np.ndarray): Inverse Jacobian matrix dE/dX.
            Q (np.ndarray): Equinoctial covariance matrix.
            QRemStat (bool): Equinoctial covariance remediation status.
            QRaw (np.ndarray): Raw equinoctial covariance matrix.
            QRem (np.ndarray): Remediated equinoctial covariance matrix.
            CRem (np.ndarray): Remediated Cartesian covariance matrix (m, m/s).
    """

    # Ensure inputs are numpy arrays
    r = np.array(r).flatten()
    v = np.array(v).flatten()
    C = np.array(C)

    # Cartestian state vector and covariance matrix
    # Pos/vel state vec. in km units
    X = np.concatenate((r, v)) / 1e3
    # Pos/vel covariance in km units
    P = C / 1e6

    def cov_make_symmetric(mat):
        return (mat + mat.T) / 2.0

    try:
        # Calculate equinoctial elements from the pos-vel state
        # convert_cartesian_to_equinoctial returns (a, n, af, ag, chi, psi, lM, F)
        # We need n, af, ag, chi, psi, lM
        _, n, af, ag, chi, psi, lM, _ = convert_cartesian_to_equinoctial(X[0:3], X[3:6], issue_warnings=False)

        if n is None:
            raise ValueError("Conversion to equinoctial elements failed (unbound or invalid orbit)")

        lM = lM % (2 * np.pi)

        # Equinoctial state vector, E
        E = np.array([n, af, ag, chi, psi, lM])

        # Jacobian matrix, J = dX/dE
        J = jacobian_equinoctial_to_cartesian(E, X)

        # Inverse of Jacobian matrix, K = dE/dX
        # J \ eye(6,6) is mathematically inv(J)
        K = np.linalg.solve(J, np.eye(6))

        # Equinoctial state covariance, Q = K * P * K'
        Q = cov_make_symmetric(K @ P @ K.T)

        # Save raw covariance
        QRaw = Q.copy()

        # Remediate eq. covariance, if required
        if RemEqCov:
            # Calc Q remediation status and remediated Q matrix
            # CovRemEigValClip returns a dict
            rem_result = CovRemEigValClip(Q)
            QRemStat = rem_result['ClipStatus']
            QRem = rem_result['Arem']

            if QRemStat:
                Q = cov_make_symmetric(QRem)
                P_rem = cov_make_symmetric(J @ Q @ J.T)
                CRem = 1e6 * P_rem
            else:
                CRem = C.copy()
        else:
            # Calc Q remediation status only
            rem_result = CovRemEigValClip(Q)
            QRemStat = rem_result['ClipStatus']
            QRem = Q.copy()
            CRem = C.copy()

    except Exception as e:
        warnings.warn(f'EquinoctialMatrices:CalculationFailure: Equinoctial state/covariance/Jacobian calculation failure. Error: {e}')
        E        = np.full(6, np.nan)
        J        = np.full((6, 6), np.nan)
        K        = np.full((6, 6), np.nan)
        Q        = np.full((6, 6), np.nan)
        QRemStat = np.nan
        QRaw     = np.full((6, 6), np.nan)
        QRem     = np.full((6, 6), np.nan)
        if RemEqCov:
            CRem = np.full((6, 6), np.nan)
        else:
            CRem = C.copy()

    return X, P, E, J, K, Q, QRemStat, QRaw, QRem, CRem
