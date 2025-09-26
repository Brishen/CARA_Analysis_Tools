"""Eigenvalue clipping remediation for covariance matrices."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict
import warnings

import numpy as np

_DEFAULT = object()
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


def _is_empty(value: Any) -> bool:
    """Return ``True`` when a MATLAB-style "empty" value is supplied."""

    if value is None:
        return True
    if isinstance(value, np.ndarray) and value.size == 0:
        return True
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return True
    return False


def _as_array(name: str, value: Any, *, ndim: int | None = None) -> np.ndarray:
    """Convert ``value`` to a ``numpy.ndarray`` and validate dimensionality."""

    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array; received shape {array.shape}")
    return array


def _serialise(value: Any) -> Any:
    """Convert numpy values into JSON-serialisable Python objects."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _serialise(val) for key, val in value.items()}
    return value


_log_file_path: Path | None = None


def _log_invocation(inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
    """Log function inputs and outputs to mirror the MATLAB implementation."""

    global _log_file_path

    try:
        json_inputs = {key: _serialise(val) for key, val in inputs.items()}
        json_outputs = {key: _serialise(val) for key, val in outputs.items()}
        log_entry = {
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "inputs": json_inputs,
            "outputs": json_outputs,
        }
        encoded = json.dumps(log_entry)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        warnings.warn(
            f"CovRemEigValClip:JSONEncodingFailed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if _log_file_path is None:
        _log_file_path = Path(__file__).with_suffix(".log")

    try:
        with _log_file_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
    except OSError as exc:  # pragma: no cover - defensive
        warnings.warn(
            f"CovRemEigValClip:LogFileOpenFailed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def CovRemEigValClip(
    Araw: Any,
    Lclip: Any = _DEFAULT,
    Lraw: Any = _DEFAULT,
    Vraw: Any = _DEFAULT,
) -> Dict[str, Any]:
    """Remediate covariance matrices by clipping their eigenvalues."""

    nargin = 1
    if Lclip is not _DEFAULT:
        nargin = 2
    if Lraw is not _DEFAULT:
        nargin = 3
    if Vraw is not _DEFAULT:
        nargin = 4

    original_inputs: Dict[str, Any] = {
        "Araw": Araw,
        "Lclip": None if Lclip is _DEFAULT else Lclip,
        "Lraw": None if Lraw is _DEFAULT else Lraw,
        "Vraw": None if Vraw is _DEFAULT else Vraw,
        "nargin": nargin,
    }

    Araw_array = _as_array("Araw", Araw, ndim=2)
    if Araw_array.shape[0] != Araw_array.shape[1]:
        raise ValueError("Araw must be a square matrix")
    if np.iscomplexobj(Araw_array):
        raise ValueError("Covariance matrix cannot have imaginary elements")

    if Lclip is _DEFAULT or _is_empty(Lclip):
        Lclip_value = 0.0
    else:
        if np.iscomplexobj(Lclip):
            raise ValueError("Lclip must be real")
        Lclip_value = float(np.asarray(Lclip))
        if Lclip_value < 0:
            raise ValueError("Lclip cannot be negative")

    if Lraw is _DEFAULT or _is_empty(Lraw):
        Lraw_array = None
    else:
        Lraw_array = _as_array("Lraw", Lraw)
        if Lraw_array.ndim != 1:
            Lraw_array = Lraw_array.reshape(-1)

    if Vraw is _DEFAULT or _is_empty(Vraw):
        Vraw_array = None
    else:
        Vraw_array = _as_array("Vraw", Vraw, ndim=2)

    if (Lraw_array is None) != (Vraw_array is None):
        raise ValueError("Lraw and Vraw must both be provided or both omitted")

    if Lraw_array is None or Vraw_array is None:
        eigvals, eigvecs = np.linalg.eigh(Araw_array)
        Lraw_array = eigvals
        Vraw_array = eigvecs
    else:
        eigvals = Lraw_array
        eigvecs = Vraw_array

    if np.iscomplexobj(Lraw_array) or np.iscomplexobj(Vraw_array):
        raise ValueError("Eigenvalues and eigenvectors must be real")

    min_eig = float(np.min(Lraw_array))
    pos_def_status = int(np.sign(min_eig))

    Lrem_array = np.array(Lraw_array, copy=True)
    clip_mask = Lrem_array < Lclip_value
    if np.any(clip_mask):
        Lrem_array[clip_mask] = Lclip_value
        clip_status = True
    else:
        clip_status = False

    Adet_value = float(np.prod(Lrem_array))

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_eigs = np.divide(
            1.0,
            Lrem_array,
            out=np.full_like(Lrem_array, np.inf, dtype=float),
            where=Lrem_array != 0,
        )
    Ainv_array = eigvecs @ np.diag(inv_eigs) @ eigvecs.T

    if clip_status:
        Arem_array = eigvecs @ np.diag(Lrem_array) @ eigvecs.T
    else:
        Arem_array = np.array(Araw_array, copy=True)

    outputs: Dict[str, Any] = {
        "Lrem": Lrem_array,
        "Lraw": Lraw_array,
        "Vraw": eigvecs,
        "PosDefStatus": pos_def_status,
        "ClipStatus": clip_status,
        "Adet": Adet_value,
        "Ainv": Ainv_array,
        "Arem": Arem_array,
        "nargout": len(_OUTPUT_ORDER),
    }

    _log_invocation(original_inputs, outputs)

    return {
        "Lrem": Lrem_array,
        "Lraw": Lraw_array,
        "Vraw": eigvecs,
        "PosDefStatus": pos_def_status,
        "ClipStatus": clip_status,
        "Adet": Adet_value,
        "Ainv": Ainv_array,
        "Arem": Arem_array,
    }
