from __future__ import annotations

import math
import pathlib
import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DistributedMatlab.ProbabilityOfCollision.Utils.CovRemEigValClip import (
    CovRemEigValClip,
)

cases = [
    {
        "id": "CovRemEigValClip-log-1",
        "inputs": {
            "Araw": [
                [
                    7.705728734620565e-21,
                    7.906493232892278e-18,
                    -5.839834542422482e-18,
                    3.0550067229847905e-18,
                    6.769423153083596e-19,
                    2.0366093404118437e-15,
                ],
                [
                    7.906493232892278e-18,
                    9.323899820546343e-13,
                    -2.570472617974921e-13,
                    1.7580052316895823e-13,
                    -7.402072430030109e-14,
                    5.024347629751825e-14,
                ],
                [
                    -5.839834542422482e-18,
                    -2.570472617974921e-13,
                    9.524049448304563e-14,
                    -4.3645812069584895e-14,
                    2.7118680456419098e-14,
                    -9.313628407014937e-13,
                ],
                [
                    3.0550067229847905e-18,
                    1.7580052316895823e-13,
                    -4.3645812069584895e-14,
                    5.5819538031593365e-14,
                    -4.151404584595468e-15,
                    4.736547797780721e-13,
                ],
                [
                    6.7694231530835935e-19,
                    -7.402072430030109e-14,
                    2.7118680456419098e-14,
                    -4.1514045845954685e-15,
                    1.5612907945049552e-13,
                    5.02159782499297e-13,
                ],
                [
                    2.0366093404118437e-15,
                    5.024347629751825e-14,
                    -9.313628407014937e-13,
                    4.736547797780722e-13,
                    5.021597824992971e-13,
                    5.430850602436654e-10,
                ],
            ],
            "Lclip": 0.0,
            "Lraw": [
                0.0,
                1.5254490230331627e-14,
                2.6907676247310066e-14,
                1.4991626279825354e-13,
                1.045020769430349e-12,
                5.430875401466847e-10,
            ],
            "Vraw": [
                [
                    0.9999999999258163,
                    -3.6741709998746196e-07,
                    7.783973074443122e-06,
                    3.3867575984095176e-06,
                    -7.880900444280854e-06,
                    3.750074504759577e-06,
                ],
                [
                    -6.852584440850214e-06,
                    -0.3168568280153361,
                    -0.05279148716619252,
                    -0.08463288420593851,
                    -0.943213695573703,
                    9.364328736075525e-05,
                ],
                [
                    7.4067210002910165e-06,
                    -0.6643014672395132,
                    -0.697843885125904,
                    -0.03562490592064297,
                    0.26541551643693717,
                    -0.0017153056911961872,
                ],
                [
                    4.65641301056322e-06,
                    0.6768623262331341,
                    -0.7078142576179992,
                    -0.09309905705450143,
                    -0.17941062541015124,
                    0.0008724005787632907,
                ],
                [
                    3.294088853774177e-06,
                    -0.012643222243246882,
                    0.09604950507215296,
                    -0.9914130389122799,
                    0.0878292236121338,
                    0.0009247974153405131,
                ],
                [
                    -3.743845187335639e-06,
                    -0.0016886150932327946,
                    -0.0006634023295533436,
                    0.0009448957299548982,
                    0.0006188895143798929,
                    0.9999977163025064,
                ],
            ],
        },
        "nargin": 4,
        "nargout": 8,
        "expected": {
            "Lrem": [
                0.0,
                1.5254490230331627e-14,
                2.6907676247310066e-14,
                1.4991626279825354e-13,
                1.045020769430349e-12,
                5.430875401466847e-10,
            ],
            "Lraw": [
                0.0,
                1.5254490230331627e-14,
                2.6907676247310066e-14,
                1.4991626279825354e-13,
                1.045020769430349e-12,
                5.430875401466847e-10,
            ],
            "Vraw": [
                [
                    0.9999999999258163,
                    -3.6741709998746196e-07,
                    7.783973074443122e-06,
                    3.3867575984095176e-06,
                    -7.880900444280854e-06,
                    3.750074504759577e-06,
                ],
                [
                    -6.852584440850214e-06,
                    -0.3168568280153361,
                    -0.05279148716619252,
                    -0.08463288420593851,
                    -0.943213695573703,
                    9.364328736075525e-05,
                ],
                [
                    7.4067210002910165e-06,
                    -0.6643014672395132,
                    -0.697843885125904,
                    -0.03562490592064297,
                    0.26541551643693717,
                    -0.0017153056911961872,
                ],
                [
                    4.65641301056322e-06,
                    0.6768623262331341,
                    -0.7078142576179992,
                    -0.09309905705450143,
                    -0.17941062541015124,
                    0.0008724005787632907,
                ],
                [
                    3.294088853774177e-06,
                    -0.012643222243246882,
                    0.09604950507215296,
                    -0.9914130389122799,
                    0.0878292236121338,
                    0.0009247974153405131,
                ],
                [
                    -3.743845187335639e-06,
                    -0.0016886150932327946,
                    -0.0006634023295533436,
                    0.0009448957299548982,
                    0.0006188895143798929,
                    0.9999977163025064,
                ],
            ],
            "PosDefStatus": 0,
            "ClipStatus": False,
            "Adet": 0.0,
            "Ainv": [
                [np.inf, -np.inf, np.inf, np.inf, np.inf, -np.inf],
                [-np.inf, np.inf, -np.inf, -np.inf, -np.inf, np.inf],
                [np.inf, -np.inf, np.inf, np.inf, np.inf, -np.inf],
                [np.inf, -np.inf, np.inf, np.inf, np.inf, -np.inf],
                [np.inf, -np.inf, np.inf, np.inf, np.inf, -np.inf],
                [-np.inf, np.inf, -np.inf, -np.inf, -np.inf, np.inf],
            ],
            "Arem": [
                [
                    7.705728734620565e-21,
                    7.906493232892278e-18,
                    -5.839834542422482e-18,
                    3.0550067229847905e-18,
                    6.769423153083596e-19,
                    2.0366093404118437e-15,
                ],
                [
                    7.906493232892278e-18,
                    9.323899820546343e-13,
                    -2.570472617974921e-13,
                    1.7580052316895823e-13,
                    -7.402072430030109e-14,
                    5.024347629751825e-14,
                ],
                [
                    -5.839834542422482e-18,
                    -2.570472617974921e-13,
                    9.524049448304563e-14,
                    -4.3645812069584895e-14,
                    2.7118680456419098e-14,
                    -9.313628407014937e-13,
                ],
                [
                    3.0550067229847905e-18,
                    1.7580052316895823e-13,
                    -4.3645812069584895e-14,
                    5.5819538031593365e-14,
                    -4.151404584595468e-15,
                    4.736547797780721e-13,
                ],
                [
                    6.7694231530835935e-19,
                    -7.402072430030109e-14,
                    2.7118680456419098e-14,
                    -4.1514045845954685e-15,
                    1.5612907945049552e-13,
                    5.02159782499297e-13,
                ],
                [
                    2.0366093404118437e-15,
                    5.024347629751825e-14,
                    -9.313628407014937e-13,
                    4.736547797780722e-13,
                    5.021597824992971e-13,
                    5.430850602436654e-10,
                ],
            ],
        },
    },
]

_OUTPUT_ORDER = (
    "Lrem",
    "Lraw",
    "Vraw",
    "PosDefStatus",
    "ClipStatus",
    "Adet",
    "Ainv",
    "Arem",
)
_INPUT_ORDER = ("Araw", "Lclip", "Lraw", "Vraw")


def _case_id(case: dict[str, Any]) -> str:
    return case.get("id", "CovRemEigValClip-case")


def _value_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)) and not value:
        return True
    if hasattr(value, "size") and getattr(value, "size", 0) == 0:
        return True
    return False


def _coerce_input(value: Any) -> Any:
    if isinstance(value, list):
        return np.array(value)
    if isinstance(value, tuple):
        return np.array(value)
    return value


def _prepare_args(case: dict[str, Any]) -> list[Any]:
    nargin = max(case.get("nargin", 0) or 0, 1)
    positional: list[Any] = []
    for index, name in enumerate(_INPUT_ORDER):
        if name == "Araw":
            positional.append(_coerce_input(case["inputs"].get(name)))
            continue
        if index >= nargin:
            break
        value = case["inputs"].get(name, None)
        if _value_is_empty(value):
            value = None
        else:
            value = _coerce_input(value)
        positional.append(value)
    return positional


def _prepare_expected(case: dict[str, Any]) -> list[Any]:
    nargout_value = case.get("nargout")
    if nargout_value is None:
        nargout = len(_OUTPUT_ORDER)
    else:
        nargout = int(nargout_value)
        if nargout == 0:
            return []
    outputs = case["expected"]
    expected = []
    for name in _OUTPUT_ORDER[:nargout]:
        expected.append(outputs.get(name, None))
    return expected


def _normalise_numeric(value: Any) -> Any:
    if isinstance(value, list):
        if not value:
            return np.array([])
        return np.array(value)
    if isinstance(value, tuple):
        return np.array(value)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
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
        if getattr(expected_arr, "dtype", None) is not None and expected_arr.dtype == bool:
            np.testing.assert_array_equal(actual_arr, expected_arr, err_msg=label)
        else:
            np.testing.assert_allclose(actual_arr, expected_arr, rtol=1e-12, atol=1e-12, err_msg=label)
        return

    if isinstance(expected, float):
        if math.isnan(expected):
            assert math.isnan(actual), f"{label} expected NaN"
            return
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12), label
        return

    assert actual == expected, f"{label} mismatch"


@pytest.mark.parametrize("case", cases, ids=_case_id)
def test_cov_rem_eig_val_clip(case: dict[str, Any]) -> None:
    positional = _prepare_args(case)
    expected_values = _prepare_expected(case)

    if not expected_values:
        CovRemEigValClip(*positional)
        return

    result = CovRemEigValClip(*positional)
    expected_len = len(expected_values)

    if isinstance(result, dict):
        actual_values = [result.get(name) for name in _OUTPUT_ORDER[:expected_len]]
    elif expected_len == 1:
        actual_values = (result,)
    else:
        actual_values = tuple(result)

    assert len(actual_values) == len(expected_values), "Output arity mismatch"

    case_id = case.get("id", "CovRemEigValClip-case")
    for name, actual, expected in zip(_OUTPUT_ORDER, actual_values, expected_values):
        _assert_equivalent(actual, expected, f"{case_id}::{name}")
