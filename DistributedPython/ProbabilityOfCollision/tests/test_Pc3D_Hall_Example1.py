import numpy as np

from DistributedPython.ProbabilityOfCollision.Pc3D_Hall_Utils.Pc3D_Hall_Example1 import run_example1


def test_pc3d_hall_example1_matches_reference_values():
    pc2d, pc3d = run_example1()

    assert np.isclose(pc2d, 1.0281653e-02, rtol=1e-7)
    assert np.isclose(pc3d, 1.0281834e-02, rtol=1e-3)
    assert np.isclose(pc2d, pc3d, rtol=1e-3)
