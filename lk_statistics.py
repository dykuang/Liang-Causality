"""Shared sufficient statistics for Torch Liang–Kleeman estimation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from numbers import Integral

import torch
from torch import Tensor


def _statistics_computation_tensor(x: Tensor) -> Tensor:
    """Promote low-precision values before LK sufficient statistics are computed."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.to(dtype=torch.float32)
    return x


@dataclass(frozen=True, eq=False)
class LKSufficientStatistics:
    """Prepared LK sufficient statistics with centered state and derivative.

    Each instance describes one batch of windows.

    ``state`` and ``derivative`` have shape ``[batch, variables, samples]``.
    The state uses all but the final ``n_step`` observations. The derivative
    uses a forward difference over ``n_step * dt`` and is centered separately.
    """

    state: Tensor
    derivative: Tensor
    n_step: int
    dt: float

    @cached_property
    def covariance(self) -> Tensor:
        """Sample covariance of the centered state, shaped ``[B, p, p]``."""
        return torch.matmul(
            self.state, self.state.transpose(-1, -2)
        ) / (self.state.shape[-1] - 1)

    @cached_property
    def cross_covariance(self) -> Tensor:
        """State/derivative cross-covariance, shaped ``[B, p, p]``."""
        return torch.matmul(
            self.state, self.derivative.transpose(-1, -2)
        ) / (self.state.shape[-1] - 1)

    @cached_property
    def covariance_ratio(self) -> Tensor:
        """Return ``H[i,j] = Cov(i,j) / Var(j)`` without cross-covariance."""
        return self.covariance_ratio_with_epsilon(torch.finfo(self.state.dtype).eps)

    def covariance_ratio_with_epsilon(self, eps: float) -> Tensor:
        """Return H using a caller-selected denominator clamp."""
        # Scale back to the state Gram matrix to match the existing Torch
        # exact-H stabilization, which clamps its diagonal at dtype eps.
        gram = self.covariance * (self.state.shape[-1] - 1)
        diagonal = torch.diagonal(gram, dim1=-2, dim2=-1)
        return gram / diagonal.unsqueeze(-2).clamp_min(eps)


def prepare_lk_statistics(
    x: Tensor, n_step: int = 1, dt: float = 1.0
) -> LKSufficientStatistics:
    """Prepare centered state and derivative from ``[B, p, T]`` windows.

    ``n_step`` and ``dt`` follow the Torch LK estimators' forward-difference
    convention. Dense covariance statistics are added lazily by the same value
    when requested by a consumer.
    """
    if not isinstance(x, Tensor) or x.ndim != 3:
        raise ValueError("x must have shape [batch, variables, time]")
    if not isinstance(n_step, Integral) or n_step < 1 or x.shape[-1] <= n_step + 1:
        raise ValueError("time dimension must be greater than n_step + 1")
    if dt <= 0:
        raise ValueError("dt must be positive")

    # Keep low-precision input values but center and differentiate in float32.
    x = _statistics_computation_tensor(x)
    n_step = int(n_step)
    history = x[..., :-n_step]
    derivative = (x[..., n_step:] - x[..., :-n_step]) / (n_step * dt)
    state = history - history.mean(dim=-1, keepdim=True)
    derivative = derivative - derivative.mean(dim=-1, keepdim=True)
    return LKSufficientStatistics(
        state=state,
        derivative=derivative,
        n_step=n_step,
        dt=float(dt),
    )


# Backwards-compatible public name.
LKStatistics = LKSufficientStatistics
