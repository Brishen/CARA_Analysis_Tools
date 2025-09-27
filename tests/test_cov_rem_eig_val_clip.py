from __future__ import annotations

import math
import pathlib
import sys
from typing import Any

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from DistributedMatlab.ProbabilityOfCollision.Utils.CovRemEigValClip import (
    CovRemEigValClip,
)

cases = [
    {
        "id": "CovRemEigValClip.log:1",
        "inputs": {
            "Araw": [
                [
                    7.7057287346205654e-21,
                    7.9064932328922779e-18,
                    -5.8398345424224816e-18,
                    3.0550067229847905e-18,
                    6.7694231530835964e-19,
                    2.0366093404118437e-15,
                ],
                [
                    7.9064932328922779e-18,
                    9.3238998205463430e-13,
                    -2.5704726179749211e-13,
                    1.7580052316895823e-13,
                    -7.4020724300301088e-14,
                    5.0243476297518248e-14,
                ],
                [
                    -5.8398345424224816e-18,
                    -2.5704726179749211e-13,
                    9.5240494483045627e-14,
                    -4.3645812069584895e-14,
                    2.7118680456419098e-14,
                    -9.3136284070149371e-13,
                ],
                [
                    3.0550067229847905e-18,
                    1.7580052316895823e-13,
                    -4.3645812069584895e-14,
                    5.5819538031593365e-14,
                    -4.1514045845954677e-15,
                    4.7365477977807213e-13,
                ],
                [
                    6.7694231530835935e-19,
                    -7.4020724300301088e-14,
                    2.7118680456419098e-14,
                    -4.1514045845954685e-15,
                    1.5612907945049552e-13,
                    5.0215978249929701e-13,
                ],
                [
                    2.0366093404118437e-15,
                    5.0243476297518248e-14,
                    -9.3136284070149371e-13,
                    4.7365477977807223e-13,
                    5.0215978249929711e-13,
                    5.4308506024366544e-10,
                ],
            ],
        },
        "nargin": 1,
        "nargout": 5,
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
            "PosDefStatus": 1,
            "ClipStatus": False,
            "Adet": [],
            "Ainv": [],
            "Arem": [],
        },
    },
    {
        "id": "CovRemEigValClip.log:2",
        "inputs": {
            "Araw": [
                [
                    4.4926594879955929e-18,
                    4.5545024217040257e-16,
                    -2.6957003888222377e-15,
                    2.6145279303962023e-16,
                    2.5249919374297476e-16,
                    1.2281026636266344e-12,
                ],
                [
                    4.5545024217040257e-16,
                    2.6693866066666511e-11,
                    -1.2077562126375471e-11,
                    8.3526920994158881e-12,
                    -1.3478652883612051e-11,
                    1.4393415546744979e-12,
                ],
                [
                    -2.6957003888222377e-15,
                    -1.2077562126375471e-11,
                    1.6602869510589600e-11,
                    -3.8731292674034249e-12,
                    -1.2222170065269484e-11,
                    -6.4305137811644886e-10,
                ],
                [
                    2.6145279303962023e-16,
                    8.3526920994158881e-12,
                    -3.8731292674034249e-12,
                    6.2801538747799553e-12,
                    -3.3898085876797321e-12,
                    1.8519461403036719e-11,
                ],
                [
                    2.5249919374297476e-16,
                    -1.3478652883612051e-11,
                    -1.2222170065269484e-11,
                    -3.3898085876797321e-12,
                    5.1257929270976439e-11,
                    6.0020676967069423e-11,
                ],
                [
                    1.2281026636266344e-12,
                    1.4393415546744979e-12,
                    -6.4305137811644886e-10,
                    1.8519461403036719e-11,
                    6.0020676967069423e-11,
                    3.3649631700523107e-07,
                ],
            ],
        },
        "nargin": 1,
        "nargout": 5,
        "expected": {
            "Lrem": [
                0.0,
                1.3865257187531055e-12,
                3.6119468297678e-12,
                3.577297836955117e-11,
                5.882268518187568e-11,
                3.3649755769234685e-07,
            ],
            "Lraw": [
                0.0,
                1.3865257187531055e-12,
                3.6119468297678e-12,
                3.577297836955117e-11,
                5.882268518187568e-11,
                3.3649755769234685e-07,
            ],
            "Vraw": [
                [
                    0.9999999997312247,
                    1.3685113309706045e-05,
                    7.689294091399022e-06,
                    -1.6582672380571235e-05,
                    1.6845983320231158e-06,
                    3.6496716978098513e-06,
                ],
                [
                    -3.7819806881726382e-06,
                    -0.5811200102887369,
                    -0.11126472065878809,
                    -0.7230107237555337,
                    0.356616304843309,
                    4.340568662864468e-06,
                ],
                [
                    1.724963951244752e-05,
                    -0.7200557170425732,
                    0.30888794293895844,
                    0.6037283425174532,
                    0.1470253000590806,
                    -0.0019111115510352422,
                ],
                [
                    -1.4507482456116746e-05,
                    0.20682685756304633,
                    0.9376851237157897,
                    -0.2586780333310695,
                    0.10514243473931098,
                    5.50571791158618e-05,
                ],
                [
                    1.4674716584645438e-06,
                    -0.31786818740169354,
                    0.11381889178206812,
                    -0.21413032001485804,
                    -0.9165987396402585,
                    0.0001784643764734931,
                ],
                [
                    -3.6161591127792306e-06,
                    -0.0013282460611433644,
                    0.0005188642810537697,
                    0.001209389493813211,
                    0.00043722600829960425,
                    0.9999981563681266,
                ],
            ],
            "PosDefStatus": 1,
            "ClipStatus": False,
            "Adet": [],
            "Ainv": [],
            "Arem": [],
        },
    },
    {
        "id": "CovRemEigValClip.log:3",
        "inputs": {
            "Araw": [
                [2.350559374029658, -5.879583266094751, -0.5743315031801796],
                [-5.879583266094751, 14.71120107459848, 1.4354301858934304],
                [-0.5743315031801796, 1.4354301858934304, 0.14100334002478163],
            ],
            "Lclip": 4.0000000000000015e-12,
        },
        "nargin": 2,
        "nargout": 7,
        "expected": {
            "Lrem": [
                0.00015316055355417182,
                0.0013293289184180684,
                17.201281299180938,
            ],
            "Lraw": [
                0.00015316055355417182,
                0.0013293289184180684,
                17.201281299180938,
            ],
            "Vraw": [
                [0.7570105694908535, 0.5388063536646807, -0.36962509510273334],
                [0.24338780972977678, 0.2924565499869817, 0.924787294703303],
                [0.6063805502768667, -0.7900359989394681, 0.09025380117004116],
            ],
            "PosDefStatus": 1,
            "ClipStatus": False,
            "Adet": 3.5021938250863114e-06,
            "Ainv": [
                [3959.99462426254, 1321.48656811451, 2676.8724410891214],
                [1321.48656811451, 451.159244304244, 789.7952570405179],
                [2676.872441089121, 789.7952570405179, 2870.2598123545604],
            ],
            "Arem": [],
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
NUMERIC_RTOL = 1e-10
NUMERIC_ATOL = 1e-10


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


def _align_matrix_sign(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    if actual.ndim != 2 or expected.ndim != 2:
        return actual
    if actual.shape != expected.shape:
        return actual

    aligned = actual.copy()
    for column in range(aligned.shape[1]):
        expected_column = expected[:, column]
        actual_column = aligned[:, column]
        if not np.any(expected_column):
            continue
        diff = np.linalg.norm(actual_column - expected_column)
        diff_neg = np.linalg.norm(-actual_column - expected_column)
        if diff_neg < diff:
            aligned[:, column] = -actual_column
    return aligned


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
            if isinstance(actual_arr, np.ndarray) and isinstance(expected_arr, np.ndarray):
                actual_arr = _align_matrix_sign(actual_arr, expected_arr)
            np.testing.assert_allclose(
                actual_arr,
                expected_arr,
                rtol=NUMERIC_RTOL,
                atol=NUMERIC_ATOL,
                err_msg=label,
            )
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
