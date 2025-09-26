#!/usr/bin/env python3
"""Convert CovRemEigValClip JSON logs into Python-friendly test vectors.

This utility consumes one or more ``CovRemEigValClip.log`` files (each emitted by
``CovRemEigValClip.m``) and produces a Python module that exposes a ``cases``
variable.  Each entry in ``cases`` preserves the inputs and outputs recorded in
the log file and captures a small amount of metadata so downstream tests can
reference the source line.

The generated output is designed to be imported directly into forthcoming unit
tests (e.g. ``from cov_rem_eig_log_cases import cases``) without needing to
perform any custom parsing within the test suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pformat
from typing import Any, Iterable, List


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the conversion utility."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert CovRemEigValClip JSON log entries into a Python module that "
            "defines a `cases` variable suitable for parametrised tests."
        )
    )
    parser.add_argument(
        "log_files",
        nargs="+",
        type=Path,
        help="Path(s) to CovRemEigValClip JSON log file(s).",
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
    """Prepare log entries for serialisation into a Python module."""

    normalised: List[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        inputs = _convert_for_python(entry.get("inputs", {}))
        outputs = _convert_for_python(entry.get("outputs", {}))
        case: dict[str, Any] = {
            "inputs": inputs,
            "expected": outputs,
        }
        if include_metadata:
            metadata = _convert_for_python(entry.get("metadata", {}))
            metadata.setdefault("timestamp", entry.get("timestamp"))
            metadata.setdefault("call_index", idx)
            case["metadata"] = metadata
        normalised.append(case)
    return normalised


def generate_module_content(variable_name: str, cases: List[dict[str, Any]]) -> str:
    """Compose the Python module content for the provided cases."""

    header = "# Auto-generated from CovRemEigValClip.log by cov_rem_eig_log_to_tests.py\n"
    header += "# Do not edit by hand.\n\n"
    body = f"{variable_name} = {pformat(cases, width=88)}\n"
    return header + body


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
