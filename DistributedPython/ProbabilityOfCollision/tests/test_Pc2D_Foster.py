import pytest
import numpy as np
from DistributedPython.ProbabilityOfCollision.Pc2D_Foster import Pc2D_Foster

class TestPc2DFoster:

    @pytest.fixture
    def omitron_test_data(self):
        r1 = np.array([378.39559, 4305.721887, 5752.767554])
        v1 = np.array([2.360800244, 5.580331936, -4.322349039])
        cov1 = np.array([
            [44.5757544811362, 81.6751751052616, -67.8687662707124],
            [81.6751751052616, 158.453402956163, -128.616921644857],
            [-67.8687662707124, -128.616921644858, 105.490542562701]
        ])

        r2 = np.array([374.5180598, 4307.560983, 5751.130418])
        v2 = np.array([-5.388125081, -3.946827739, 3.322820358])
        cov2 = np.array([
            [2.31067077720423, 1.69905293875632, -1.4170164577661],
            [1.69905293875632, 1.24957388457206, -1.04174164279599],
            [-1.4170164577661, -1.04174164279599, 0.869260558223714]
        ])

        OmitronHBR = 0.020
        Tol = 1e-09
        Accuracy = 0.001

        return {
            'r1': r1, 'v1': v1, 'cov1': cov1,
            'r2': r2, 'v2': v2, 'cov2': cov2,
            'HBR': OmitronHBR, 'Tol': Tol, 'Accuracy': Accuracy
        }

    def test_circular_hbr(self, omitron_test_data):
        data = omitron_test_data
        expSolution = 2.70601573490125e-05

        Pc, Arem, IsPosDef, IsRemediated = Pc2D_Foster(
            data['r1'], data['v1'], data['cov1'],
            data['r2'], data['v2'], data['cov2'],
            data['HBR'], data['Tol'], 'circle'
        )

        assert Pc == pytest.approx(expSolution, rel=data['Accuracy'])

    def test_square_hbr(self, omitron_test_data):
        data = omitron_test_data
        expSolution = 3.44534649703584e-05

        Pc, Arem, IsPosDef, IsRemediated = Pc2D_Foster(
            data['r1'], data['v1'], data['cov1'],
            data['r2'], data['v2'], data['cov2'],
            data['HBR'], data['Tol'], 'square'
        )

        assert Pc == pytest.approx(expSolution, rel=data['Accuracy'])

    def test_eq_area_square_hbr(self, omitron_test_data):
        data = omitron_test_data
        expSolution = 2.70601573490111e-05

        Pc, Arem, IsPosDef, IsRemediated = Pc2D_Foster(
            data['r1'], data['v1'], data['cov1'],
            data['r2'], data['v2'], data['cov2'],
            data['HBR'], data['Tol'], 'squareEquArea'
        )

        assert Pc == pytest.approx(expSolution, rel=data['Accuracy'])

    def test_invalid_hbr_type(self, omitron_test_data):
        data = omitron_test_data

        with pytest.raises(ValueError, match="ellipse as HBRType is not supported"):
            Pc2D_Foster(
                data['r1'], data['v1'], data['cov1'],
                data['r2'], data['v2'], data['cov2'],
                data['HBR'], data['Tol'], 'ellipse'
            )
