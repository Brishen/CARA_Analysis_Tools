"""Coppola (2012b) conjunction time bounds for linear encounters."""

from __future__ import annotations

import math
from typing import Any, Iterable, Tuple

import numpy as np

try:  # pragma: no cover - prefer fast native implementation when available
    _NUMPY_ERFCINV = np.erfcinv  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - NumPy 2.0 removed ``erfcinv``
    _NUMPY_ERFCINV = None

try:  # pragma: no cover - high-precision fallback
    import mpmath
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _MPMATH = None
else:  # pragma: no cover - configure precision once
    _MPMATH = mpmath
    _MPMATH.mp.dps = max(_MPMATH.mp.dps, 128)


_MIN_GAMMA = np.float64('1.0000000498663174e-16')
_MAX_GAMMA = 2.0


def _erfcinv_scalar(value: float) -> float:
    """Return the inverse complementary error function for a scalar value."""

    if math.isnan(value):
        return math.nan
    if value <= 0.0:
        return math.inf
    if value >= 2.0:
        return -math.inf
    if value == 1.0:
        return 0.0

    if value < 1.0:
        estimate = math.sqrt(-math.log(value / 2.0))
    else:
        estimate = -math.sqrt(-math.log((2.0 - value) / 2.0))

    two_over_sqrt_pi = 2.0 / math.sqrt(math.pi)
    for _ in range(8):
        residual = math.erfc(estimate) - value
        if abs(residual) <= 1.0e-15 * max(1.0, abs(value)):
            break
        derivative = -two_over_sqrt_pi * math.exp(-estimate * estimate)
        estimate -= residual / derivative

    return estimate


def _erfcinv(values: np.ndarray) -> np.ndarray:
    """Return the element-wise inverse complementary error function."""

    array = np.asarray(values, dtype=float)
    if _NUMPY_ERFCINV is not None:
        return _NUMPY_ERFCINV(array)

    result = np.empty_like(array)
    if _MPMATH is not None:
        it = np.nditer(
            [array, result],
            flags=["refs_ok", "multi_index"],
            op_flags=[["readonly"], ["writeonly"]],
        )
        for input_value, output_slot in it:
            value = float(input_value)
            if value <= 0.0:
                output_slot[...] = math.inf
            elif value >= 2.0:
                output_slot[...] = -math.inf
            elif value == 1.0:
                output_slot[...] = 0.0
            else:
                mp_value = _MPMATH.mpf(value)
                if mp_value < 1:
                    estimate = _MPMATH.sqrt(-_MPMATH.log(mp_value / 2))
                else:
                    estimate = -_MPMATH.sqrt(-_MPMATH.log((2 - mp_value) / 2))

                for _ in range(20):
                    residual = _MPMATH.erfc(estimate) - mp_value
                    if _MPMATH.almosteq(residual, 0, rel_eps=_MPMATH.mpf('1e-40')):
                        break
                    derivative = -2 / _MPMATH.sqrt(_MPMATH.pi) * _MPMATH.exp(-estimate**2)
                    estimate -= residual / derivative

                output_slot[...] = float(estimate)
        return result

    it = np.nditer([array, result], flags=["refs_ok", "multi_index"], op_flags=[["readonly"], ["writeonly"]])
    for input_value, output_slot in it:
        output_slot[...] = _erfcinv_scalar(float(input_value))
    return result


def _as_vector(name: str, value: Any) -> np.ndarray:
    """Convert an input vector to a 1D array of length three."""

    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 3:
        raise ValueError(f"{name} must contain exactly three elements")
    return array


def conj_bounds_Coppola(
    gamma: Any,
    HBR: Any,
    rci: Iterable[float],
    vci: Iterable[float],
    Pci: Any,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Calculate conjunction time bounds using Coppola's formulation."""

    gamma_array = np.asarray(gamma, dtype=float)
    gamma_shape = gamma_array.shape
    tau0 = np.full(gamma_shape, np.nan, dtype=float)
    tau1 = np.full(gamma_shape, np.nan, dtype=float)

    if np.any(gamma_array < 0.0) or np.any(gamma_array > _MAX_GAMMA):
        raise ValueError("gamma must lie in the inclusive range [0, 2]")

    safe_gamma = np.where(gamma_array <= 0.0, _MIN_GAMMA, gamma_array)

    HBR_value = float(np.asarray(HBR, dtype=float))
    rci_vec = _as_vector("rci", rci)
    vci_vec = _as_vector("vci", vci)

    Pci_array = np.asarray(Pci, dtype=float)
    if Pci_array.ndim != 2 or Pci_array.shape[0] < 3 or Pci_array.shape[1] < 3:
        raise ValueError("Pci must be at least a 3x3 matrix")
    Aci = Pci_array[:3, :3]

    v0mag = np.linalg.norm(vci_vec)
    if v0mag < 100.0 * np.finfo(float).eps:
        tau0.fill(-np.inf)
        tau1.fill(np.inf)
        return tau0, tau1, float(-np.inf), float(np.inf)

    xhat = vci_vec / v0mag
    yhat = rci_vec - xhat * np.dot(xhat, rci_vec)
    yhat_norm = np.linalg.norm(yhat)
    if yhat_norm < np.finfo(float).eps:
        raise ValueError("Relative position and velocity vectors are colinear")
    yhat /= yhat_norm
    zhat = np.cross(xhat, yhat)

    eROTi = np.vstack((xhat, yhat, zhat))

    rce = eROTi @ rci_vec
    Ace = eROTi @ Aci @ eROTi.T

    eta2 = Ace[0, 0]
    w = Ace[1:3, 0]
    Pc = Ace[1:3, 1:3]
    b = np.linalg.solve(Pc.T, w)
    sv2 = np.sqrt(max(0.0, 2.0 * (eta2 - float(b @ w))))

    q0 = float(b @ rce[1:3])
    bTb = float(b @ b)
    dmin = -HBR_value * np.sqrt(max(0.0, bTb))
    dmax = HBR_value * np.sqrt(max(0.0, 1.0 + bTb))

    q0_minus_dmax = q0 - dmax
    q0_minus_dmin = q0 - dmin

    tau0_gam1 = q0_minus_dmax / v0mag
    tau1_gam1 = q0_minus_dmin / v0mag

    ac = _erfcinv(safe_gamma)
    temp = ac * sv2
    tau0[...] = (-temp + q0_minus_dmax) / v0mag
    tau1[...] = (temp + q0_minus_dmin) / v0mag

    return tau0, tau1, float(tau0_gam1), float(tau1_gam1)
