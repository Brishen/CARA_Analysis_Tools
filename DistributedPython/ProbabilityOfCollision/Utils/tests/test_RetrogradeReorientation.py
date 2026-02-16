import pytest
import numpy as np
from DistributedPython.ProbabilityOfCollision.Utils.RetrogradeReorientation import RetrogradeReorientation, CheckForRetrograde

def test_CheckForRetrograde():
    r = np.array([1, 0, 0])
    v = np.array([0, 1, 0])
    # h = [0, 0, 1]. Retro = 1 + 1 < Eps (False)
    assert not CheckForRetrograde(r, v, 1e-6)

    r = np.array([1, 0, 0])
    v = np.array([0, -1, 0])
    # h = [0, 0, -1]. Retro = 1 - 1 < Eps (True)
    assert CheckForRetrograde(r, v, 1e-6)

def test_RetrogradeReorientation_NoReorientation():
    r1 = np.array([1, 0, 0])
    v1 = np.array([0, 1, 0])
    C1 = np.eye(6)
    r2 = np.array([2, 0, 0])
    v2 = np.array([0, 2, 0])
    C2 = np.eye(6)

    params = {'RetrogradeReorientation': 0}
    r1_out, v1_out, C1_out, r2_out, v2_out, C2_out, out = RetrogradeReorientation(r1, v1, C1, r2, v2, C2, params)

    assert not out['Reoriented']
    np.testing.assert_array_equal(r1, r1_out.flatten())

def test_RetrogradeReorientation_RetrogradeOrbit():
    # Make primary retrograde
    r1 = np.array([10000, 0, 0])
    v1 = np.array([0, -7, 0]) # Retrograde equatorial
    C1 = np.eye(6)
    r2 = np.array([7000, 0, 0])
    v2 = np.array([0, 7.5, 0])
    C2 = np.eye(6)

    params = {'RetrogradeReorientation': 1}
    r1_out, v1_out, C1_out, r2_out, v2_out, C2_out, out = RetrogradeReorientation(r1, v1, C1, r2, v2, C2, params)

    assert out['Reoriented']
    assert out['Retro1']
    assert not out['Retro2']

    # Check that new primary is not retrograde
    assert not CheckForRetrograde(r1_out, v1_out, 1e-6)

def test_RetrogradeReorientation_ForceRetrograde():
    r1 = np.array([10000, 0, 0])
    v1 = np.array([0, 7, 0]) # Prograde
    C1 = np.eye(6)
    r2 = np.array([7000, 0, 0])
    v2 = np.array([0, 7.5, 0])
    C2 = np.eye(6)

    params = {'RetrogradeReorientation': 3}
    r1_out, v1_out, C1_out, r2_out, v2_out, C2_out, out = RetrogradeReorientation(r1, v1, C1, r2, v2, C2, params)

    assert out['Reoriented']
    # Check if primary is now retrograde
    assert CheckForRetrograde(r1_out, v1_out, 1e-6)

def test_RetrogradeReorientation_InputShapes():
    r1 = np.array([[1], [0], [0]])
    v1 = np.array([[0], [1], [0]])
    C1 = np.eye(6)
    r2 = np.array([[2], [0], [0]])
    v2 = np.array([[0], [2], [0]])
    C2 = np.eye(6)

    params = {'RetrogradeReorientation': 0}
    r1_out, v1_out, C1_out, r2_out, v2_out, C2_out, out = RetrogradeReorientation(r1, v1, C1, r2, v2, C2, params)

    assert r1_out.shape == (3, 1)

    r1 = np.array([1, 0, 0])
    v1 = np.array([0, 1, 0])
    r2 = np.array([2, 0, 0])
    v2 = np.array([0, 2, 0])

    r1_out, v1_out, C1_out, r2_out, v2_out, C2_out, out = RetrogradeReorientation(r1, v1, C1, r2, v2, C2, params)

    assert r1_out.shape == (3,)

def test_RetrogradeReorientation_InvalidParam():
    r1 = np.array([1, 0, 0])
    v1 = np.array([0, 1, 0])
    C1 = np.eye(6)
    r2 = np.array([2, 0, 0])
    v2 = np.array([0, 2, 0])
    C2 = np.eye(6)

    params = {'RetrogradeReorientation': 99}
    with pytest.raises(ValueError):
        RetrogradeReorientation(r1, v1, C1, r2, v2, C2, params)
