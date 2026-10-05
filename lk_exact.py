"""Graph-only exact Torch Liang–Kleeman estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from lk_statistics import LKSufficientStatistics, _statistics_computation_tensor


@dataclass(frozen=True, eq=False)
class ExactGraphResult:
    """Exact graph outputs and the dtype used for their computation.

    The tensors have shape ``[batch, variables, variables]``. ``effective_dtype``
    records the dtype used by the covariance calculations and linear solve.
    """

    graph: Tensor
    coefficients: Tensor
    covariance_ratio: Tensor
    effective_dtype: torch.dtype


def estimate_exact_graph(
    statistics: LKSufficientStatistics,
    *,
    singular_fallback: Literal["pinv", "solve"] = "pinv",
) -> ExactGraphResult:
    """Estimate the exact LK graph from prepared sufficient statistics.

    The coefficient matrix is ``solve(C, D)`` and the graph is the
    elementwise product of that matrix with the covariance ratio. A batched
    ``solve_ex`` attempts every window independently; windows with a nonzero
    status use the selected fallback. The default ``pinv`` fallback keeps one
    singular covariance from interrupting well-conditioned windows in the batch.

    Half and bfloat16 statistics are promoted to float32 before dense
    statistics are materialized. The result reports that effective dtype.
    """
    if singular_fallback not in {"pinv", "solve"}:
        raise ValueError("singular_fallback must be 'pinv' or 'solve'")
    if not isinstance(statistics, LKSufficientStatistics):
        raise TypeError("statistics must be an LKSufficientStatistics value")

    state = statistics.state
    derivative = statistics.derivative
    if not isinstance(state, Tensor) or not isinstance(derivative, Tensor):
        raise TypeError("LK sufficient statistics state and derivative must be tensors")
    if state.ndim != 3 or derivative.shape != state.shape:
        raise ValueError("LK sufficient statistics must have matching [batch, variables, samples] tensors")
    if state.shape[-1] < 2:
        raise ValueError("at least two samples are required for exact LK estimation")
    if state.device != derivative.device or state.dtype != derivative.dtype:
        raise ValueError("LK sufficient statistics state and derivative must share device and dtype")
    if not state.is_floating_point():
        raise TypeError("LK sufficient statistics must use a floating-point dtype")

    state = _statistics_computation_tensor(state)
    derivative = _statistics_computation_tensor(derivative)
    effective_dtype = state.dtype
    if effective_dtype != statistics.state.dtype:
        statistics = LKSufficientStatistics(
            state=state,
            derivative=derivative,
            n_step=statistics.n_step,
            dt=statistics.dt,
        )

    covariance = statistics.covariance
    cross_covariance = statistics.cross_covariance
    covariance_ratio = statistics.covariance_ratio

    coefficients, info = torch.linalg.solve_ex(
        covariance, cross_covariance, check_errors=False
    )
    failed = info != 0
    if failed.any():
        if singular_fallback == "pinv":
            fallback = torch.linalg.pinv(covariance[failed]) @ cross_covariance[failed]
        else:
            fallback = torch.linalg.solve(
                covariance[failed], cross_covariance[failed]
            )
        coefficients = coefficients.clone()
        coefficients[failed] = fallback

    graph = covariance_ratio * coefficients
    return ExactGraphResult(
        graph=graph,
        coefficients=coefficients,
        covariance_ratio=covariance_ratio,
        effective_dtype=effective_dtype,
    )
