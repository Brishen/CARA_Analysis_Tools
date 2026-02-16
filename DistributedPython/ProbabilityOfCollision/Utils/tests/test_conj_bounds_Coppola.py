import numpy as np
import pytest
from DistributedPython.ProbabilityOfCollision.Utils.conj_bounds_Coppola import conj_bounds_Coppola

def test_conj_bounds_Coppola_basic():
    # Construct a simple case.
    # rci = [0, 100, 0]
    # vci = [10, 0, 0]
    # Pci = identity
    # HBR = 10
    # gamma = 0.01 (some small probability threshold)

    rci = np.array([0, 100, 0])
    vci = np.array([10, 0, 0])
    Pci = np.eye(3)
    HBR = 10.0
    gamma = 1e-16

    tau0, tau1, t0_g1, t1_g1 = conj_bounds_Coppola(gamma, HBR, rci, vci, Pci)

    assert tau0 < tau1
    assert not np.isnan(tau0)
    assert not np.isnan(tau1)

def test_conj_bounds_Coppola_vector_gamma():
    rci = np.array([0, 100, 0])
    vci = np.array([10, 0, 0])
    Pci = np.eye(3)
    HBR = 10.0
    gamma = np.array([1e-16, 1e-10])

    tau0, tau1, t0_g1, t1_g1 = conj_bounds_Coppola(gamma, HBR, rci, vci, Pci)

    assert tau0.shape == gamma.shape
    assert tau1.shape == gamma.shape
    # erfcinv(small) is large.
    # temp = ac * sv2. large * positive.
    # tau0 = (-temp + ...) -> smaller (more negative).
    # So tau0 should decrease as gamma decreases.
    # 1e-16 < 1e-10. So tau0[0] < tau0[1].
    assert tau0[0] < tau0[1]

def test_conj_bounds_Coppola_zero_velocity():
    rci = np.array([0, 100, 0])
    vci = np.array([0, 0, 0])
    Pci = np.eye(3)
    HBR = 10.0
    gamma = 1e-16

    tau0, tau1, t0_g1, t1_g1 = conj_bounds_Coppola(gamma, HBR, rci, vci, Pci)

    assert np.isinf(tau0)
    assert np.isinf(tau1)

def test_conj_bounds_Coppola_parallel_r_v():
    # Special case where r and v are parallel (yhat construction)
    rci = np.array([100, 0, 0])
    vci = np.array([10, 0, 0])
    Pci = np.eye(3)
    HBR = 10.0
    gamma = 1e-16

    tau0, tau1, t0_g1, t1_g1 = conj_bounds_Coppola(gamma, HBR, rci, vci, Pci)

    assert not np.isnan(tau0)
    assert not np.isnan(tau1)
