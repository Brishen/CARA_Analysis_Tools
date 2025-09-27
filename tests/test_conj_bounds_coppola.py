# Auto-generated from conj_bounds_Coppola.log by conj_bounds_log_to_tests.py
# Do not edit by hand.


from __future__ import annotations

from typing import Any

import numpy as np
import pytest

try:
    from DistributedMatlab.ProbabilityOfCollision.Utils.conj_bounds_Coppola import (
        conj_bounds_Coppola,
    )
except ModuleNotFoundError:  # pragma: no cover - graceful skip until available
    conj_bounds_Coppola = None

cases = [{'expected': {'tau0': -28.466684015003157,
               'tau0_gam1': -26.766586631461234,
               'tau1': -24.61750019759193,
               'tau1_gam1': -26.317597581133853},
  'id': 'conj_bounds.log:1',
  'inputs': {'HBR': 20,
             'Pci': [[2307548.332236509,
                      -5849932.525175953,
                      -379366.0137657096,
                      -762.0286790694754,
                      -739.9292727525119,
                      6440.899625033573],
                     [-5849932.525175953,
                      14834977.446265582,
                      960449.8604731602,
                      1933.3144971931208,
                      1878.6117949248787,
                      -16331.682383246682],
                     [-379366.0137657096,
                      960449.8604731602,
                      63059.90898582124,
                      125.14948923875737,
                      120.64626957894163,
                      -1058.4832607963435],
                     [-762.0286790694754,
                      1933.3144971931208,
                      125.14948923875737,
                      0.25334809642908657,
                      0.24547510494688182,
                      -2.1288386783060274],
                     [-739.9292727525119,
                      1878.6117949248787,
                      120.64626957894163,
                      0.24547510494688182,
                      0.23927793670976158,
                      -2.0671124083292614],
                     [6440.899625033573,
                      -16331.682383246682,
                      -1058.4832607963435,
                      -2.1288386783060274,
                      -2.0671124083292614,
                      17.98126788997221]],
             'gamma': 0,
             'rci': [-4278.047994919121, 5921.425147906179, 116.0172589160502],
             'vci': [-51.24106699999993, -36.58255000000099, -22.33212900000001],
             'verbose': False},
  'nargin': 6,
  'nargout': 4}]

_OUTPUT_ORDER = ('tau0', 'tau1', 'tau0_gam1', 'tau1_gam1')
_INPUT_ORDER = ('gamma', 'HBR', 'rci', 'vci', 'Pci', 'verbose')


def _case_id(case: dict[str, Any]) -> str:
    return case.get('id', 'conj_bounds_Coppola-case')


def _value_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)) and not value:
        return True
    if hasattr(value, 'size') and getattr(value, 'size', 0) == 0:
        return True
    return False


def _coerce_input(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return np.array(value)
    return value


def _prepare_args(case: dict[str, Any]) -> list[Any]:
    nargin = max(case.get('nargin', 0) or 0, 1)
    positional: list[Any] = []
    for index, name in enumerate(_INPUT_ORDER):
        if index >= nargin:
            break
        value = case['inputs'].get(name, None)
        if _value_is_empty(value):
            value = None
        else:
            value = _coerce_input(value)
        positional.append(value)
    return positional


def _prepare_expected(case: dict[str, Any]) -> list[Any]:
    nargout_value = case.get('nargout')
    if nargout_value is None:
        nargout = len(_OUTPUT_ORDER)
    else:
        nargout = int(nargout_value)
        if nargout == 0:
            return []
    outputs = case['expected']
    expected = []
    for name in _OUTPUT_ORDER[:nargout]:
        expected.append(outputs.get(name, None))
    return expected


def _normalise_numeric(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            return np.array([])
        return np.array(value)
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return np.array(value)
    return value


def _assert_equivalent(actual: Any, expected: Any, label: str) -> None:
    if _value_is_empty(expected):
        assert _value_is_empty(actual), f"{label} expected empty value"
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{label} expected dict, got {type(actual).__name__}"
        assert actual.keys() == expected.keys(), f"{label} keys mismatch"
        for key in expected:
            _assert_equivalent(actual[key], expected[key], f"{label}.{key}")
        return

    if isinstance(expected, (list, tuple)):
        actual_arr = _normalise_numeric(actual)
        expected_arr = _normalise_numeric(expected)
        np.testing.assert_allclose(actual_arr, expected_arr, rtol=1e-12, atol=1e-12, err_msg=label)
        return

    if isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12), label
        return

    assert actual == expected, f"{label} mismatch"


@pytest.mark.parametrize('case', cases, ids=_case_id)
def test_conj_bounds_coppola(case: dict[str, Any]) -> None:
    if conj_bounds_Coppola is None:
        pytest.skip('conj_bounds_Coppola Python implementation is not yet available.')

    positional = _prepare_args(case)
    expected_values = _prepare_expected(case)

    if not expected_values:
        conj_bounds_Coppola(*positional)
        return

    result = conj_bounds_Coppola(*positional)
    expected_len = len(expected_values)

    if isinstance(result, dict):
        actual_values = [result.get(name) for name in _OUTPUT_ORDER[:expected_len]]
    elif expected_len == 1:
        actual_values = (result,)
    else:
        actual_values = tuple(result)

    assert len(actual_values) == len(expected_values), 'Output arity mismatch'

    case_id = case.get('id', 'conj_bounds_Coppola-case')
    for name, actual, expected in zip(_OUTPUT_ORDER, actual_values, expected_values):
        _assert_equivalent(actual, expected, f"{case_id}::{name}")
