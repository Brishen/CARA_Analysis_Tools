import unittest

import numpy as np

from DistributedPython.ProbabilityOfCollision.Pc3D_Hall_Utils.Pc3D_Hall_Example1 import Pc3D_Hall_Example1


class TestPc3D_Hall_Example1(unittest.TestCase):

    def test_example1_values(self):
        pc2d, pc3d = Pc3D_Hall_Example1()

        self.assertTrue(np.isclose(pc2d, 1.0281653e-02, rtol=1e-6))
        self.assertTrue(np.isclose(pc3d, 1.0281834e-02, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
