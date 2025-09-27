"""Coppola (2012b) conjunction time bounds for linear encounters."""

from __future__ import annotations

from typing import Any, Iterable, Tuple

import numpy as np

try:  # pragma: no cover - runtime feature detection
    _ERFINV = np.erfinv  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - numpy < 1.17
    try:  # pragma: no cover - optional SciPy dependency
        from scipy.special import erfinv as _ERFINV  # type: ignore
    except ImportError as exc:  # pragma: no cover - provide clear guidance
        raise ImportError(
            "conj_bounds_Coppola requires numpy.erfinv or scipy.special.erfinv"
        ) from exc


def _erfcinv(values: np.ndarray) -> np.ndarray:
    """Return the element-wise inverse complementary error function."""

    return _ERFINV(1.0 - values)


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

    if verbose:
        print()
        print(
            "Calculating conjunction time bounds using the Coppola (2012b) formulation:"
        )

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

    ac = _erfcinv(gamma_array)
    temp = ac * sv2
    tau0[:] = (-temp + q0_minus_dmax) / v0mag
    tau1[:] = (temp + q0_minus_dmin) / v0mag

    if verbose:
        flat_gamma = gamma_array.reshape(-1)
        flat_tau0 = tau0.reshape(-1)
        flat_tau1 = tau1.reshape(-1)
        for g_val, t0_val, t1_val in zip(flat_gamma, flat_tau0, flat_tau1):
            print(
                f"  gamma = {g_val}  tau0 = {t0_val}  tau1 = {t1_val}"
            )

    return tau0, tau1, float(tau0_gam1), float(tau1_gam1)
