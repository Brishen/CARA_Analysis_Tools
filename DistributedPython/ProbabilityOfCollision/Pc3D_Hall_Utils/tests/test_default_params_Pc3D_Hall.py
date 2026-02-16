import numpy as np
import pytest
from DistributedPython.ProbabilityOfCollision.Pc3D_Hall_Utils.default_params_Pc3D_Hall import default_params_Pc3D_Hall

def test_default_params_Pc3D_Hall_basics():
    params = default_params_Pc3D_Hall()

    assert params['gamma'] == 1e-16
    assert params['Neph'] == 101
    assert params['use_Lebedev'] is True
    assert params['deg_Lebedev'] == 5810

    # Check Lebedev vectors and weights loaded
    assert 'vec_Lebedev' in params
    assert 'wgt_Lebedev' in params
    assert params['vec_Lebedev'].shape == (3, 5810)
    assert params['wgt_Lebedev'].shape == (5810,)

def test_default_params_Pc3D_Hall_custom():
    input_params = {'gamma': 1e-10, 'Neph': 201, 'deg_Lebedev': 6}
    params = default_params_Pc3D_Hall(input_params)

    assert params['gamma'] == 1e-10
    assert params['Neph'] == 201
    assert params['deg_Lebedev'] == 6
    assert params['vec_Lebedev'].shape == (3, 6)
    assert params['wgt_Lebedev'].shape == (6,)

def test_default_params_Pc3D_Hall_invalid_Texpand():
    with pytest.raises(ValueError, match='Invalid Texpand parameter'):
        default_params_Pc3D_Hall({'Texpand': -1})

def test_default_params_Pc3D_Hall_Tmin_limit_invalid():
    with pytest.raises(ValueError, match='Invalid Tmin_limit and/or Tmax_limit parameters'):
        default_params_Pc3D_Hall({'Tmin_limit': 100, 'Tmax_limit': 50}) # min > max
