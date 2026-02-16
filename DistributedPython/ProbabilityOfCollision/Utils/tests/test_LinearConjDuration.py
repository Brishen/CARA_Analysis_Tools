import numpy as np
import pytest
from DistributedPython.ProbabilityOfCollision.Utils.LinearConjDuration import LinearConjDuration

def test_LinearConjDuration_simple():
    r1 = np.array([0, 0, 0])
    v1 = np.array([1, 0, 0])
    cov1 = np.eye(3)
    r2 = np.array([10, 1, 0])
    v2 = np.array([-1, 0, 0])
    cov2 = np.eye(3)
    HBR = 1.0

    # params default
    tau0, tau1, dtau, taum, delt = LinearConjDuration(r1, v1, cov1, r2, v2, cov2, HBR)

    # Check that intervals are calculated
    assert not np.isnan(tau0)
    assert not np.isnan(tau1)
    assert tau0 < tau1
    assert dtau == tau1 - tau0
    assert taum == (tau1 + tau0) / 2
    assert delt == max(dtau, abs(tau0), abs(tau1))

def test_LinearConjDuration_FindCA():
    # Setup a case where objects are not at TCA
    # t = -5. Objects at t=-5 relative to TCA.
    # r1(0) = (0,0,0), v1 = (1,0,0) -> r1(-5) = (-5, 0, 0)
    # r2(0) = (10,1,0), v2 = (-1,0,0) -> r2(-5) = (15, 1, 0)

    r1 = np.array([-5, 0, 0])
    v1 = np.array([1, 0, 0])
    cov1 = np.eye(3)
    r2 = np.array([15, 1, 0])
    v2 = np.array([-1, 0, 0])
    cov2 = np.eye(3)
    HBR = 1.0

    params = {'FindCA': True}

    tau0, tau1, dtau, taum, delt = LinearConjDuration(r1, v1, cov1, r2, v2, cov2, HBR, params)

    # The function should internally propagate to TCA.
    # The result (tau0, tau1) are relative to TCA.
    # So they should be roughly symmetric around 0 if the encounter is symmetric.

    assert not np.isnan(tau0)
    assert abs(taum) < 1.0 # Centered near TCA (0)
