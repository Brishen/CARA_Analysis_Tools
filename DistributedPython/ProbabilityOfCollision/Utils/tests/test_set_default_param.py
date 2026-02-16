import pytest
from DistributedPython.ProbabilityOfCollision.Utils.set_default_param import set_default_param

def test_set_default_param_basic():
    params = {}
    params = set_default_param(params, 'test', 1)
    assert params['test'] == 1

def test_set_default_param_existing():
    params = {'test': 2}
    params = set_default_param(params, 'test', 1)
    assert params['test'] == 2

def test_set_default_param_none_value():
    params = {'test': None}
    params = set_default_param(params, 'test', 1)
    assert params['test'] == 1

def test_set_default_param_none_params():
    params = None
    params = set_default_param(params, 'test', 1)
    assert params == {'test': 1}

def test_set_default_param_invalid_name():
    with pytest.raises(ValueError):
        set_default_param({}, '', 1)
    with pytest.raises(ValueError):
        set_default_param({}, None, 1)
