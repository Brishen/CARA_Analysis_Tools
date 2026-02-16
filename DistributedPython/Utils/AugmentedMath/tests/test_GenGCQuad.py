import unittest
import numpy as np
from DistributedPython.Utils.AugmentedMath.GenGCQuad import GenGCQuad

class TestGenGCQuad(unittest.TestCase):
    def test_n16(self):
        xGC, yGC, wGC = GenGCQuad(16)
        self.assertEqual(xGC.shape, (1, 16))
        self.assertEqual(yGC.shape, (1, 16))
        self.assertEqual(wGC.shape, (1, 16))

        # Check first row of table (approx)
        # -9.829730996839018e-01     1.837495178165702e-01     6.773407894247465e-03
        self.assertAlmostEqual(xGC[0, 0], -9.829730996839018e-01, places=15)
        self.assertAlmostEqual(yGC[0, 0], 1.837495178165702e-01, places=15)
        self.assertAlmostEqual(wGC[0, 0], 6.773407894247465e-03, places=15)

        # Check last row
        # 9.829730996839018e-01     1.837495178165702e-01     6.773407894247465e-03
        self.assertAlmostEqual(xGC[0, -1], 9.829730996839018e-01, places=15)

    def test_n64(self):
        xGC, yGC, wGC = GenGCQuad(64)
        self.assertEqual(xGC.shape, (1, 64))

        # Check first row
        # -9.988322268323265e-01     4.831337952550822e-02     4.657833967754515e-04
        self.assertAlmostEqual(xGC[0, 0], -9.988322268323265e-01, places=15)

    def test_n_other(self):
        NGC = 10
        xGC, yGC, wGC = GenGCQuad(NGC)
        self.assertEqual(xGC.shape, (1, NGC))

        # Verify formula consistency
        # xGC = cos(vGC) where vGC = pi/(N+1) * [N, ..., 1]
        cGC = np.pi / (NGC + 1)
        vGC = cGC * np.arange(NGC, 0, -1)
        expected_x = np.cos(vGC)
        np.testing.assert_allclose(xGC.flatten(), expected_x, rtol=1e-15)

if __name__ == '__main__':
    unittest.main()
