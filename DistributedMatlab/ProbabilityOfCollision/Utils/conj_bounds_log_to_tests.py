#!/usr/bin/env python3
"""Convert ``conj_bounds_Coppola`` JSON logs into pytest regression tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pformat
import textwrap
from typing import Any, Iterable, List

OUTPUT_ORDER = ("tau0", "tau1", "tau0_gam1", "tau1_gam1")
INPUT_ORDER = ("gamma", "HBR", "rci", "vci", "Pci", "verbose")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the conversion utility."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert conj_bounds_Coppola JSON log entries into a parametrised "
            "pytest module that replays the captured MATLAB invocations."
        )
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        type=Path,
        help="Path(s) to conj_bounds_Coppola JSON log file(s).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output path for the generated Python module. The module is "
            "printed to stdout when omitted."
        ),
    )
    parser.add_argument(
        "--variable-name",
        default="cases",
        help="Name of the top-level variable that will hold the parsed cases.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help=(
            "Retain metadata such as timestamps and originating log file names. "
            "The metadata is included by default; this flag is provided to keep "
            "CLI symmetry if tests ever prefer to drop it."
        ),
    )
    parser.add_argument(
        "--exclude-metadata",
        dest="include_metadata",
        action="store_false",
        help="Omit metadata from the generated output.",
    )
    parser.set_defaults(include_metadata=True)
    return parser.parse_args()


def load_log_entries(log_paths: Iterable[Path]) -> List[dict[str, Any]]:
    """Load JSON objects from each provided log file."""

    entries: List[dict[str, Any]] = []
    for log_path in log_paths:
        if not log_path.exists():
            raise FileNotFoundError(f"Log file does not exist: {log_path}")
        if log_path.is_dir():
            raise IsADirectoryError(f"Expected a file path, received directory: {log_path}")

        with log_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {log_path} on line {line_number}: {exc.msg}"
                    ) from exc
                entry.setdefault("metadata", {})
                entry["metadata"].update(
                    {"log_path": str(log_path), "line_number": line_number}
                )
                entries.append(entry)
    return entries


def _convert_for_python(value: Any) -> Any:
    """Recursively convert JSON-native values for deterministic Python output."""

    if isinstance(value, list):
        return [_convert_for_python(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _convert_for_python(val) for key, val in value.items()}
    return value


def normalise_entries(
    entries: Iterable[dict[str, Any]], include_metadata: bool
) -> List[dict[str, Any]]:
    """Prepare log entries for serialisation into a pytest module."""

    normalised: List[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        inputs = _convert_for_python(entry.get("inputs", {}))
        outputs = _convert_for_python(entry.get("outputs", {}))

        nargin = int(inputs.pop("nargin", 0) or 0)
        nargout = int(outputs.pop("nargout", 0) or 0)
        effective_verbose = outputs.pop("effectiveVerbose", None)

        metadata = _convert_for_python(entry.get("metadata", {}))
        metadata.setdefault("timestamp", entry.get("timestamp"))
        metadata.setdefault("call_index", idx)
        if effective_verbose is not None:
            metadata.setdefault("effectiveVerbose", effective_verbose)

        log_path = metadata.get("log_path")
        line_number = metadata.get("line_number")
        if log_path and line_number:
            case_id = f"{Path(log_path).name}:{line_number}"
        else:
            case_id = f"case_{idx}"

        case: dict[str, Any] = {
            "id": case_id,
            "inputs": inputs,
            "nargin": nargin,
            "nargout": nargout,
            "expected": outputs,
        }

        if include_metadata and metadata:
            case["metadata"] = metadata

        normalised.append(case)

    return normalised


def generate_module_content(variable_name: str, cases: List[dict[str, Any]]) -> str:
    """Compose the pytest module content for the provided cases."""

    header = "# Auto-generated from conj_bounds_Coppola.log by conj_bounds_log_to_tests.py\n"
    header += "# Do not edit by hand.\n\n"

    cases_literal = pformat(cases, width=88)

    helpers = textwrap.dedent(
        f"""
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

{variable_name} = {cases_literal}

_OUTPUT_ORDER = {OUTPUT_ORDER!r}
_INPUT_ORDER = {INPUT_ORDER!r}


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
        assert _value_is_empty(actual), f"{{label}} expected empty value"
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{{label}} expected dict, got {{type(actual).__name__}}"
        assert actual.keys() == expected.keys(), f"{{label}} keys mismatch"
        for key in expected:
            _assert_equivalent(actual[key], expected[key], f"{{label}}.{{key}}")
        return

    if isinstance(expected, (list, tuple)):
        actual_arr = _normalise_numeric(actual)
        expected_arr = _normalise_numeric(expected)
        np.testing.assert_allclose(actual_arr, expected_arr, rtol=1e-12, atol=1e-12, err_msg=label)
        return

    if isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12), label
        return

    assert actual == expected, f"{{label}} mismatch"


@pytest.mark.parametrize('case', {variable_name}, ids=_case_id)
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
        _assert_equivalent(actual, expected, f"{{case_id}}::{{name}}")
"""
    )
    return header + helpers


def main() -> None:
    args = parse_args()
    entries = load_log_entries(args.log_files)
    cases = normalise_entries(entries, include_metadata=args.include_metadata)
    module_content = generate_module_content(args.variable_name, cases)

    if args.output is None:
        print(module_content)
    else:
        args.output.write_text(module_content, encoding="utf-8")


if __name__ == "__main__":
    main()
