"""A common selection interface for exact and amortized LK estimation."""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn

from Causality_est_torch import LiangCausalityEstimator
from amortized_lk import AmortizedLK


class LKResult(NamedTuple):
    """Common outputs; exact-only uncertainty fields are ``None`` for learned LK."""

    graph: Tensor
    variance: Optional[Tensor]
    normalized_graph: Optional[Tensor]
    coefficients: Optional[Tensor]
    covariance_ratio: Optional[Tensor]


class LKEstimator(nn.Module):
    """Select either the existing exact estimator or learned LK approximation.

    Args:
        mode: ``"exact"`` or ``"amortized"``.
        amortized_options: keyword options forwarded to :class:`AmortizedLK`.
        amortized_model: optionally provide an already constructed/trained model.

    Both modes accept ``[batch, variables, time]`` and return :class:`LKResult`.
    The exact mode preserves the legacy variance and normalized-graph outputs;
    the amortized mode returns predicted coefficients and covariance ratios and
    has no variance estimate.
    """

    def __init__(
        self,
        mode: str = "exact",
        *,
        n_step: int = 1,
        dt: float = 1.0,
        amortized_options: Optional[dict] = None,
        amortized_model: Optional[AmortizedLK] = None,
    ):
        super().__init__()
        if mode not in {"exact", "amortized"}:
            raise ValueError("mode must be 'exact' or 'amortized'")
        self.mode = mode
        self.n_step = n_step
        self.dt = dt
        if mode == "exact":
            self.estimator = LiangCausalityEstimator(n_step=n_step, dt=dt)
        else:
            if amortized_model is not None and amortized_options:
                raise ValueError("provide amortized_model or amortized_options, not both")
            self.estimator = amortized_model or AmortizedLK(
                n_step=n_step, dt=dt, **(amortized_options or {})
            )

    def forward(
        self,
        x: Tensor,
        *,
        projection: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> LKResult:
        if self.mode == "exact":
            graph, variance, normalized_graph = self.estimator(x)
            return LKResult(graph, variance, normalized_graph, None, None)
        graph, coefficients, ratio = self.estimator(
            x, projection=projection, generator=generator
        )
        return LKResult(graph, None, None, coefficients, ratio)


def load_amortized_checkpoint(
    path: str,
    *,
    map_location: str | torch.device = "cpu",
    projection_rank: Optional[int] = None,
) -> AmortizedLK:
    """Restore an amortized model from ``train_amortized_lk.py`` output."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    config = checkpoint["config"]
    model = AmortizedLK(
        width=config.get("width", 32),
        projection_rank=projection_rank or config.get("projection_rank", 64),
        n_step=config.get("n_step", 1),
        dt=config.get("dt", 1.0),
        h_method=config.get("h_method", "projection"),
        encoder_type=config.get("encoder_type", "tcn"),
        decoder_type=config.get("decoder_type", "pair_mlp"),
        use_global_context=config.get("use_global_context", True),
        low_rank=config.get("low_rank", 16),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model
