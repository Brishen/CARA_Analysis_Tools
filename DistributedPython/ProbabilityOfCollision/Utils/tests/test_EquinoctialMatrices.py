import numpy as np
import pytest
from DistributedPython.ProbabilityOfCollision.Utils.EquinoctialMatrices import EquinoctialMatrices

def test_EquinoctialMatrices_valid_orbit():
    # Define a simple LEO orbit
    # r = 7000 km, v = 7.5 km/s (approx circular)
    r_km = np.array([7000.0, 0.0, 0.0])
    v_km = np.array([0.0, 7.5, 0.0])

    # Convert to meters
    r_m = r_km * 1000.0
    v_m = v_km * 1000.0

    # Define a simple covariance (diagonal, 1 km^2 pos, 1e-6 km^2/s^2 vel)
    # P_km = diag([1, 1, 1, 1e-6, 1e-6, 1e-6])
    P_km = np.diag([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6])

    # Convert to meters^2 and (m/s)^2
    # Pos cov: km^2 -> m^2 (* 1e6)
    # Vel cov: (km/s)^2 -> (m/s)^2 (* 1e6)
    C_m = P_km * 1e6

    # Test without remediation
    X, P, E, J, K, Q, QRemStat, QRaw, QRem, CRem = EquinoctialMatrices(r_m, v_m, C_m, RemEqCov=False)

    # Check shapes
    assert X.shape == (6,)
    assert P.shape == (6, 6)
    assert E.shape == (6,)
    assert J.shape == (6, 6)
    assert K.shape == (6, 6)
    assert Q.shape == (6, 6)
    assert QRaw.shape == (6, 6)
    assert QRem.shape == (6, 6)
    assert CRem.shape == (6, 6)

    # Check values
    # X should match input (in km)
    assert np.allclose(X, np.concatenate((r_km, v_km)))

    # P should match input (in km)
    assert np.allclose(P, P_km)

    # E elements:
    # a approx 7000 km
    # n approx sqrt(mu/a^3)
    mu = 3.986004418e5
    a = 1.0 / (2.0/np.linalg.norm(r_km) - np.linalg.norm(v_km)**2/mu)
    n_expected = np.sqrt(mu / a**3)
    assert np.isclose(E[0], n_expected, rtol=1e-3)

    # Q should be symmetric
    assert np.allclose(Q, Q.T)

def test_EquinoctialMatrices_remediation():
    # Use the same orbit
    r_m = np.array([7000000.0, 0.0, 0.0])
    v_m = np.array([0.0, 7500.0, 0.0])

    # Create a covariance that might have negative eigenvalues in equinoctial space
    # It's hard to guarantee without specific values, but we can check the plumbing.
    # We'll use a valid covariance first.
    P_km = np.diag([1.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6])
    C_m = P_km * 1e6

    # Test with remediation
    X, P, E, J, K, Q, QRemStat, QRaw, QRem, CRem = EquinoctialMatrices(r_m, v_m, C_m, RemEqCov=True)

    # Since P is PD, Q should be PD (mostly), so QRemStat might be False (no clipping needed)
    # But if it was clipped, QRemStat would be True.
    # The important thing is that the function runs and returns correct types.
    assert isinstance(QRemStat, bool) or isinstance(QRemStat, np.bool_)
    assert QRem.shape == (6, 6)

    if QRemStat:
        # If clipped, CRem should be updated
        assert not np.array_equal(CRem, C_m)
    else:
        # If not clipped, CRem should be same as C
        assert np.array_equal(CRem, C_m)

def test_EquinoctialMatrices_failure():
    # Invalid orbit (r=0)
    r_m = np.array([0.0, 0.0, 0.0])
    v_m = np.array([0.0, 0.0, 0.0])
    C_m = np.eye(6)

    # Expect NaNs and warning
    with pytest.warns(UserWarning, match='EquinoctialMatrices:CalculationFailure'):
        X, P, E, J, K, Q, QRemStat, QRaw, QRem, CRem = EquinoctialMatrices(r_m, v_m, C_m, RemEqCov=False)

    assert np.all(np.isnan(E))
    assert np.all(np.isnan(J))
    assert np.all(np.isnan(K))
    assert np.all(np.isnan(Q))
    assert np.isnan(QRemStat)

def test_EquinoctialMatrices_shapes():
    # Test handling of 1x3 vs 3x1 inputs
    r_m = np.array([[7000000.0, 0.0, 0.0]]) # 1x3
    v_m = np.array([[0.0], [7500.0], [0.0]]) # 3x1
    C_m = np.eye(6) * 1e6

    X, P, E, J, K, Q, QRemStat, QRaw, QRem, CRem = EquinoctialMatrices(r_m, v_m, C_m, RemEqCov=False)

    assert X.shape == (6,)
    assert not np.any(np.isnan(X))
