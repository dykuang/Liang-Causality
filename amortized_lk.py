"""Label-free, amortized Liang–Kleeman coefficient operator.

This module predicts the multivariate regression coefficients used by the LK
estimator and trains them from matrix-free stochastic normal-equation
residuals. It deliberately does not call or cache the exact LK solution.

Inputs use ``[batch, variables, time]``. Coefficients use the convention
``W[i, j]`` = coefficient of source variable ``i`` in the derivative of
target variable ``j``; the causal matrix is ``H * W`` with
``H[i, j] = Cov(i, j) / Var(j)``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from lk_statistics import LKSufficientStatistics, prepare_lk_statistics


LKInput = Tensor | LKSufficientStatistics


def _normalization_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _normalization_epsilon(dtype: torch.dtype) -> float:
    return torch.finfo(_normalization_dtype(dtype)).eps


def _as_lk_sufficient_statistics(
    x: LKInput, n_step: int, dt: float
) -> LKSufficientStatistics:
    """Reuse prepared statistics or prepare a raw window.

    ``n_step`` and ``dt`` apply only when ``x`` is a raw window. An
    ``LKSufficientStatistics`` input is reused as-is and retains its own ``n_step`` and
    ``dt``.
    """
    if isinstance(x, LKSufficientStatistics):
        return x
    return prepare_lk_statistics(x, n_step=n_step, dt=dt)


def _state_and_derivative(
    x: LKInput, n_step: int, dt: float
) -> tuple[Tensor, Tensor]:
    """Return centered state and differences as ``[B,p,m]``.

    ``n_step`` and ``dt`` configure preparation for a raw window. If ``x`` is
    an ``LKSufficientStatistics`` input, its prepared state and derivative are reused
    unchanged with its own ``n_step`` and ``dt``.
    """
    statistics = _as_lk_sufficient_statistics(x, n_step, dt)
    return statistics.state, statistics.derivative


def stochastic_probe_residual_loss(
    x: LKInput,
    w_hat: Tensor,
    *,
    q: int = 4,
    ridge: Optional[float] = None,
    ridge_scale: float = 1e-3,
    dt: float = 1.0,
    n_step: int = 1,
    normalized: bool = False,
    loss_mode: Optional[str] = None,
    probe_distribution: str = "rademacher",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Estimate the ridge normal-equation residual loss without forming C,D.

    Fresh Rademacher probes ``v`` satisfy ``E[v v.T] = I``, so the unnormalized
    objective is an unbiased estimate of
    ``||(C + lambda I) W_hat - D||_F**2``. The optional normalized variant
    divides each probe residual by its corresponding ``||D v||**2`` and is
    useful as a scale-balanced training objective, but is not an unbiased
    estimate of the unnormalized Frobenius objective.

    ``C`` and ``D`` are never materialized. The only data-dependent matrix
    products are with the window-shaped state and derivative arrays.

    ``n_step`` and ``dt`` configure preparation when ``x`` is a raw window.
    An ``LKSufficientStatistics`` input is reused as-is with its own ``n_step`` and
    ``dt``.
    """
    if q < 1:
        raise ValueError("q must be at least 1")
    z, y = _state_and_derivative(x, n_step, dt)
    batch, variables, m = z.shape
    if w_hat.shape != (batch, variables, variables):
        raise ValueError("w_hat must have shape [batch, variables, variables]")
    w_hat = w_hat.to(dtype=z.dtype)

    if ridge is None:
        # tr(C)/p from the sample variance diagonal, without forming C.
        mean_variance = z.square().sum(dim=(-1, -2)) / (variables * (m - 1))
        lam = ridge_scale * mean_variance
    else:
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        lam = z.new_full((batch,), ridge)

    if probe_distribution == "rademacher":
        probes = torch.empty(
            batch, variables, q, device=z.device, dtype=z.dtype
        ).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
    elif probe_distribution == "gaussian":
        probes = torch.randn(
            batch, variables, q, device=z.device, dtype=z.dtype,
            generator=generator,
        )
    elif probe_distribution == "orthogonal":
        if q > variables:
            raise ValueError("orthogonal probes require q <= number of variables")
        gaussian = torch.randn(
            batch, variables, q, device=z.device, dtype=z.dtype,
            generator=generator,
        )
        probes = torch.linalg.qr(gaussian, mode="reduced").Q * math.sqrt(variables)
    else:
        raise ValueError("probe_distribution must be rademacher, gaussian, or orthogonal")

    wv = w_hat @ probes
    # C(Wv)-Dv = Z.T @ (Z @ Wv - Y @ v) / (m - 1).
    residual_samples = z @ (
        z.transpose(1, 2) @ wv - y.transpose(1, 2) @ probes
    ) / (m - 1)
    residual_samples = residual_samples + lam[:, None, None] * wv
    probe_loss = residual_samples.square().sum(dim=1)
    mode = loss_mode or ("normalized" if normalized else "plain")
    if mode == "normalized":
        dv = z @ (y.transpose(1, 2) @ probes) / (m - 1)
        scale = dv.square().sum(dim=1).clamp_min(torch.finfo(z.dtype).eps)
        probe_loss = probe_loss / scale
    elif mode == "jacobi":
        diagonal = z.square().sum(dim=-1) / (m - 1)
        preconditioner = (diagonal + lam[:, None]).clamp_min(
            torch.finfo(z.dtype).eps
        ).rsqrt()
        probe_loss = (residual_samples * preconditioner.unsqueeze(-1)).square().sum(dim=1)
    elif mode != "plain":
        raise ValueError("loss_mode must be plain, normalized, or jacobi")
    return probe_loss.mean()


def random_projection_covariance_ratio(
    x: LKInput,
    *,
    rank: int = 64,
    n_step: int = 1,
    dt: float = 1.0,
    eps: Optional[float] = None,
    projection: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Estimate ``H[i,j]=Cov(i,j)/Var(j)`` using a temporal sketch.

    Supply a fixed ``[rank, time-n_step]`` projection to make inference
    deterministic for repeated windows of the same length. Otherwise a fresh
    Gaussian projection is drawn on each call. The output is ``[B,p,p]``.

    ``n_step`` and ``dt`` configure preparation when ``x`` is a raw window.
    An ``LKSufficientStatistics`` input is reused as-is with its own ``n_step`` and
    ``dt``.
    """
    if rank < 1:
        raise ValueError("rank must be at least 1")
    z, _ = _state_and_derivative(x, n_step, dt)
    batch, variables, m = z.shape
    if projection is None:
        projection = torch.randn(
            rank, m, dtype=z.dtype, device=z.device, generator=generator
        ) / math.sqrt(rank)
    else:
        if projection.shape != (rank, m):
            raise ValueError("projection must have shape [rank, time-n_step]")
        projection = projection.to(device=z.device, dtype=z.dtype)
    sketch = z @ projection.transpose(0, 1)
    gram = sketch @ sketch.transpose(1, 2)
    diagonal = torch.diagonal(gram, dim1=-2, dim2=-1)
    if eps is None:
        eps = torch.finfo(z.dtype).eps
    return gram / diagonal.unsqueeze(-2).clamp_min(eps)


def exact_covariance_ratio(
    x: LKInput, *, n_step: int = 1, dt: float = 1.0, eps: Optional[float] = None
) -> Tensor:
    """Compute the exact sample ``H`` ratio for evaluation-only H0 baselines.

    ``n_step`` and ``dt`` configure preparation when ``x`` is a raw window.
    An ``LKSufficientStatistics`` input is reused as-is with its own ``n_step`` and
    ``dt``.
    """
    statistics = _as_lk_sufficient_statistics(x, n_step, dt)
    if eps is None:
        return statistics.covariance_ratio
    return statistics.covariance_ratio_with_epsilon(eps)


def _sampled_covariance_ratio_targets(
    statistics: LKSufficientStatistics, pairs: Tensor, eps: Optional[float]
) -> tuple[Tensor, Tensor]:
    """Return selected H targets and normalized pair indices."""
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [num_pairs, 2]")
    z = statistics.state
    pairs = pairs.to(device=z.device, dtype=torch.long)
    i, j = pairs.unbind(dim=1)
    numerator = (z[:, i] * z[:, j]).sum(dim=-1)
    denominator = z[:, j].square().sum(dim=-1)
    if eps is None:
        eps = torch.finfo(z.dtype).eps
    return numerator / denominator.clamp_min(eps), pairs


def sampled_covariance_ratio_loss(
    x: LKInput, h_hat: Tensor, pairs: Tensor, *, n_step: int = 1,
    dt: float = 1.0, eps: Optional[float] = None,
) -> Tensor:
    """Supervise selected H[i,j] ratios without an LK label or full H.

    ``n_step`` and ``dt`` configure preparation when ``x`` is a raw window.
    An ``LKSufficientStatistics`` input is reused as-is with its own ``n_step``
    and ``dt``.
    """
    statistics = _as_lk_sufficient_statistics(x, n_step, dt)
    batch, variables, _ = statistics.state.shape
    if h_hat.shape != (batch, variables, variables):
        raise ValueError("h_hat must have shape [batch, variables, variables]")
    target, pairs = _sampled_covariance_ratio_targets(statistics, pairs, eps)
    i, j = pairs.unbind(dim=1)
    prediction = h_hat[:, i, j]
    return (prediction - target).square().mean()


def sampled_covariance_ratio_pair_loss(
    x: LKInput, h_pairs: Tensor, pairs: Tensor, *, n_step: int = 1,
    dt: float = 1.0, eps: Optional[float] = None,
) -> Tensor:
    """Supervise selected covariance ratios without materializing full H.

    ``n_step`` and ``dt`` configure preparation when ``x`` is a raw window.
    An ``LKSufficientStatistics`` input is reused as-is with its own ``n_step``
    and ``dt``.
    """
    statistics = _as_lk_sufficient_statistics(x, n_step, dt)
    target, _ = _sampled_covariance_ratio_targets(statistics, pairs, eps)
    if h_pairs.shape != (statistics.state.shape[0], target.shape[1]):
        raise ValueError("h_pairs must have shape [batch, num_pairs]")
    return (h_pairs - target).square().mean()


class SharedTemporalEncoder(nn.Module):
    """Three-layer temporal CNN applied independently to each variable."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, variables, time = x.shape
        encoded = self.network(x.reshape(batch * variables, 1, time)).squeeze(-1)
        return encoded.reshape(batch, variables, -1)


class DownsampledMLPEncoder(nn.Module):
    def __init__(self, width: int, bins: int = 32):
        super().__init__()
        self.bins = bins
        self.network = nn.Sequential(
            nn.Linear(bins, 2 * width), nn.GELU(),
            nn.Linear(2 * width, width), nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, variables, time = x.shape
        pooled = F.adaptive_avg_pool1d(
            x.reshape(batch * variables, 1, time), self.bins
        ).flatten(1)
        return self.network(pooled).reshape(batch, variables, -1)


class SpectralEncoder(nn.Module):
    def __init__(self, width: int, bins: int = 32):
        super().__init__()
        self.bins = bins
        self.network = nn.Sequential(
            nn.Linear(2 * bins, 2 * width), nn.GELU(),
            nn.Linear(2 * width, width), nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, variables, time = x.shape
        flattened = x.reshape(batch * variables, time)
        spectrum = torch.fft.rfft(flattened, dim=-1, norm="ortho")
        components = torch.stack((spectrum.real, spectrum.imag), dim=1)
        pooled = F.adaptive_avg_pool1d(components, self.bins).flatten(1)
        return self.network(pooled).reshape(batch, variables, -1)


class AttentionEncoder(nn.Module):
    def __init__(self, width: int, heads: int = 4, max_tokens: int = 64):
        super().__init__()
        if width % heads:
            raise ValueError("attention encoder width must be divisible by 4")
        self.max_tokens = max_tokens
        self.input_projection = nn.Linear(1, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=2 * width,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.attention = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: Tensor) -> Tensor:
        batch, variables, time = x.shape
        sequence = F.adaptive_avg_pool1d(
            x.reshape(batch * variables, 1, time), self.max_tokens
        ).transpose(1, 2)
        length = sequence.shape[1]
        positions = torch.arange(length, device=x.device, dtype=x.dtype)
        half = (self.input_projection.out_features + 1) // 2
        frequencies = torch.exp(
            torch.arange(half, device=x.device, dtype=x.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = positions[:, None] * frequencies[None, :]
        position_encoding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        position_encoding = position_encoding[:, : self.input_projection.out_features]
        tokens = self.input_projection(sequence) + position_encoding[None, :, :]
        encoded = self.attention(tokens).mean(dim=1)
        return encoded.reshape(batch, variables, -1)


class BilinearEdgeDecoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.bilinear = nn.Parameter(torch.empty(width, width))
        nn.init.xavier_uniform_(self.bilinear)
        self.source = nn.Linear(width, 1, bias=False)
        self.target = nn.Linear(width, 1, bias=False)
        self.context = nn.Linear(width, 1)

    def forward(self, source: Tensor, target: Tensor, context: Tensor) -> Tensor:
        batch, variables, _, width = source.shape
        context_edges = self.context(context)[:, None, None, :]
        bilinear = (source @ self.bilinear * target).sum(dim=-1) / math.sqrt(width)
        return (
            bilinear + self.source(source).squeeze(-1)
            + self.target(target).squeeze(-1) + context_edges.squeeze(-1)
        )


class LowRankEdgeDecoder(nn.Module):
    def __init__(self, width: int, rank: int):
        super().__init__()
        self.source = nn.Linear(2 * width, rank)
        self.target = nn.Linear(2 * width, rank)

    def forward(self, h: Tensor, global_context: Tensor) -> Tensor:
        variables = h.shape[1]
        g = global_context[:, None, :].expand(-1, variables, -1)
        source = self.source(torch.cat((h, g), dim=-1))
        target = self.target(torch.cat((h, g), dim=-1))
        return source @ target.transpose(1, 2) / math.sqrt(source.shape[-1])


class LowRankCorrelationHead(nn.Module):
    """PSD low-rank correlation head with cheap selected-pair scoring."""

    def __init__(self, width: int, rank: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or width
        self.network = nn.Sequential(
            nn.Linear(2 * width, hidden), nn.GELU(), nn.Linear(hidden, rank)
        )

    def factors(self, h: Tensor, global_context: Tensor) -> Tensor:
        context = global_context[:, None, :].expand(-1, h.shape[1], -1)
        return self.network(torch.cat((h, context), dim=-1))

    @staticmethod
    def _scale_ratio(numerator: Tensor, denominator: Tensor, eps: float) -> Tensor:
        denominator = torch.where(
            denominator.abs() < eps, denominator.new_full((), eps), denominator
        )
        return numerator / denominator

    @staticmethod
    def _normalization_factors(factors: Tensor) -> tuple[Tensor, torch.dtype, float]:
        output_dtype = factors.dtype
        compute_dtype = _normalization_dtype(output_dtype)
        normalized_factors = factors.to(dtype=compute_dtype)
        return normalized_factors, output_dtype, torch.finfo(compute_dtype).eps

    @staticmethod
    def score_pairs(factors: Tensor, scales: Tensor, pairs: Tensor, eps: float) -> Tensor:
        pairs = pairs.to(device=factors.device, dtype=torch.long)
        i, j = pairs.unbind(dim=1)
        normalized_factors, output_dtype, normalization_eps = (
            LowRankCorrelationHead._normalization_factors(factors)
        )
        u_i, u_j = normalized_factors[:, i], normalized_factors[:, j]
        rho = (u_i * u_j).sum(dim=-1)
        rho = rho / (u_i.square().sum(dim=-1) * u_j.square().sum(dim=-1)).sqrt().clamp_min(normalization_eps)
        scale_ratio = LowRankCorrelationHead._scale_ratio(
            scales[:, i], scales[:, j], eps
        )
        return (rho * scale_ratio).to(dtype=output_dtype)

    @staticmethod
    def correlation(factors: Tensor) -> Tensor:
        normalized_factors, output_dtype, eps = (
            LowRankCorrelationHead._normalization_factors(factors)
        )
        gram = normalized_factors @ normalized_factors.transpose(1, 2)
        diagonal = torch.diagonal(gram, dim1=-2, dim2=-1)
        rho = gram / (diagonal.unsqueeze(-1) * diagonal.unsqueeze(-2)).sqrt().clamp_min(eps)
        return rho.to(dtype=output_dtype)

    @staticmethod
    def materialize(factors: Tensor, scales: Tensor, eps: float) -> Tensor:
        rho = LowRankCorrelationHead.correlation(factors)
        scale_ratio = LowRankCorrelationHead._scale_ratio(
            scales.unsqueeze(-1), scales.unsqueeze(-2), eps
        )
        return rho * scale_ratio


def _make_encoder(encoder_type: str, width: int) -> nn.Module:
    encoders = {
        "tcn": lambda: SharedTemporalEncoder(width),
        "downsample_mlp": lambda: DownsampledMLPEncoder(width),
        "spectral": lambda: SpectralEncoder(width),
        "attention": lambda: AttentionEncoder(width),
    }
    if encoder_type not in encoders:
        raise ValueError(f"unknown encoder_type {encoder_type!r}; choose {tuple(encoders)}")
    return encoders[encoder_type]()


class AmortizedLK(nn.Module):
    """Permutation-equivariant, label-free approximation of the LK graph.

    ``exact`` computes only the covariance-ratio matrix, never the LK solve.
    ``learned_low_rank`` constructs a PSD low-rank correlation matrix and
    rescales it with exact per-channel RMS values.
    """

    _H_ALIASES = {"learned": "learned_pair"}
    _H_METHODS = {"exact", "projection", "learned_pair", "learned_low_rank"}

    def __init__(
        self, width: int = 32, projection_rank: int = 64, n_step: int = 1,
        dt: float = 1.0, h_method: str = "projection", encoder_type: str = "tcn",
        decoder_type: str = "pair_mlp", use_global_context: bool = True,
        low_rank: int = 16, h_rank: int = 8, h_factor_hidden: Optional[int] = None,
        h_structure: str = "correlation_psd",
    ):
        super().__init__()
        if width < 1 or projection_rank < 1 or low_rank < 1:
            raise ValueError("width, projection_rank, and low_rank must be at least 1")
        h_method = self._H_ALIASES.get(h_method, h_method)
        if h_method not in self._H_METHODS:
            raise ValueError(f"unknown h_method {h_method!r}; choose {sorted(self._H_METHODS)}")
        if h_method == "learned_low_rank" and h_rank < 1:
            raise ValueError("h_rank must be at least 1 for learned_low_rank H")
        if h_structure != "correlation_psd":
            raise ValueError("only h_structure='correlation_psd' is currently supported")
        self.encoder = _make_encoder(encoder_type, width)
        self.encoder_type, self.decoder_type = encoder_type, decoder_type
        decoder_types = {"pair_mlp", "bilinear", "low_rank", "bilinear_residual"}
        if decoder_type not in decoder_types:
            raise ValueError(f"unknown decoder_type {decoder_type!r}; choose {sorted(decoder_types)}")
        self.use_global_context, self.low_rank = use_global_context, low_rank
        self.global_phi = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.global_rho = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.edge_decoder = nn.Sequential(
            nn.Linear(5 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width),
            nn.GELU(), nn.Linear(width, 1),
        ) if decoder_type in {"pair_mlp", "bilinear_residual"} else None
        self.bilinear_decoder = BilinearEdgeDecoder(width) if decoder_type in {"bilinear", "bilinear_residual"} else None
        self.low_rank_decoder = LowRankEdgeDecoder(width, low_rank) if decoder_type == "low_rank" else None
        self.projection_rank, self.n_step, self.dt = projection_rank, n_step, dt
        self.h_method, self.h_rank, self.h_structure = h_method, h_rank, h_structure
        self.h_decoder = nn.Sequential(
            nn.Linear(5 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width),
            nn.GELU(), nn.Linear(width, 1),
        ) if h_method == "learned_pair" else None
        self.low_rank_h_head = LowRankCorrelationHead(width, h_rank, h_factor_hidden) if h_method == "learned_low_rank" else None

    def _features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, variables, time]")
        compute_dtype = _normalization_dtype(x.dtype)
        normalization_input = x.to(dtype=compute_dtype)
        centered = normalization_input - normalization_input.mean(dim=-1, keepdim=True)
        rms = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
            torch.finfo(compute_dtype).eps
        )
        anchor = centered.gather(-1, centered.abs().argmax(dim=-1, keepdim=True))
        signed_scales = rms * torch.where(
            anchor < 0, -torch.ones_like(anchor), torch.ones_like(anchor)
        )
        normalized = (centered / signed_scales).to(dtype=x.dtype)
        h = self.encoder(normalized)
        context = self.global_rho(self.global_phi(h).mean(dim=1)) if self.use_global_context else torch.zeros_like(h.mean(dim=1))
        return h, context, signed_scales.squeeze(-1).to(dtype=x.dtype), centered

    def _pair_features(
        self, h: Tensor, context: Tensor,
        source: Optional[Tensor] = None, target: Optional[Tensor] = None,
    ) -> Tensor:
        variables = h.shape[1]
        source = source if source is not None else h.unsqueeze(2).expand(-1, -1, variables, -1)
        target = target if target is not None else h.unsqueeze(1).expand(-1, variables, -1, -1)
        global_edges = context[:, None, None, :].expand(-1, variables, variables, -1)
        return torch.cat((source, target, source * target, source - target, global_edges), dim=-1)

    def _coefficients(self, h: Tensor, context: Tensor, scales: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        variables = h.shape[1]
        source = h.unsqueeze(2).expand(-1, -1, variables, -1)
        target = h.unsqueeze(1).expand(-1, variables, -1, -1)
        needs_pair = self.decoder_type in {"pair_mlp", "bilinear_residual"} or self.h_method == "learned_pair"
        pair_features = self._pair_features(h, context, source, target) if needs_pair else None
        if self.decoder_type == "pair_mlp":
            w_normalized = self.edge_decoder(pair_features).squeeze(-1)
        elif self.decoder_type == "bilinear":
            w_normalized = self.bilinear_decoder(source, target, context)
        elif self.decoder_type == "low_rank":
            w_normalized = self.low_rank_decoder(h, context)
        else:
            w_normalized = self.bilinear_decoder(source, target, context) + self.edge_decoder(pair_features).squeeze(-1)
        return w_normalized * scales.unsqueeze(1) / scales.unsqueeze(2), pair_features

    def predict_coefficients(self, x: Tensor) -> Tensor:
        """Predict W only, avoiding all covariance-ratio work."""
        h, context, scales, _ = self._features(x)
        return self._coefficients(h, context, scales)[0]

    def _predict_covariance_ratio(
        self,
        x: Tensor,
        *,
        projection: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
        feature_state: Optional[tuple[Tensor, Tensor, Tensor]] = None,
        pair_features: Optional[Tensor] = None,
    ) -> Tensor:
        if self.h_method == "exact":
            return exact_covariance_ratio(x, n_step=self.n_step, dt=self.dt)
        if self.h_method == "projection":
            return random_projection_covariance_ratio(
                x, rank=self.projection_rank, n_step=self.n_step, dt=self.dt,
                projection=projection, generator=generator,
            )

        if feature_state is None:
            h, context, scales, _ = self._features(x)
        else:
            h, context, scales = feature_state
        if self.h_method == "learned_pair":
            if pair_features is None:
                pair_features = self._pair_features(h, context)
            return self.h_decoder(pair_features).squeeze(-1)
        factors = self.low_rank_h_head.factors(h, context)
        return self.low_rank_h_head.materialize(
            factors, scales, _normalization_epsilon(x.dtype)
        )

    def predict_h_correlation(self, x: Tensor) -> Tensor:
        """Predict the normalized correlation used by learned low-rank H."""
        if self.h_method != "learned_low_rank":
            raise ValueError("correlation prediction is available for learned_low_rank H only")
        h, context, _, _ = self._features(x)
        factors = self.low_rank_h_head.factors(h, context)
        return self.low_rank_h_head.correlation(factors)

    def predict_covariance_ratio(
        self,
        x: Tensor,
        *,
        projection: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Predict H alone without running the coefficient decoder for W."""
        return self._predict_covariance_ratio(
            x, projection=projection, generator=generator
        )

    def score_h_pairs(self, x: Tensor, pairs: Tensor) -> Tensor:
        """Score selected H pairs without materializing the dense H matrix."""
        if self.h_method != "learned_low_rank":
            raise ValueError("selected-pair scoring is available for learned_low_rank H only")
        h, context, scales, _ = self._features(x)
        factors = self.low_rank_h_head.factors(h, context)
        return self.low_rank_h_head.score_pairs(
            factors, scales, pairs, _normalization_epsilon(x.dtype)
        )

    def forward(self, x: Tensor, *, projection: Optional[Tensor] = None,
                generator: Optional[torch.Generator] = None, compute_graph: bool = True
    ) -> tuple[Optional[Tensor], Tensor, Optional[Tensor]]:
        h, context, scales, _ = self._features(x)
        w_hat, pair_features = self._coefficients(h, context, scales)
        if not compute_graph:
            return None, w_hat, None
        h_hat = self._predict_covariance_ratio(
            x,
            projection=projection,
            generator=generator,
            feature_state=(h, context, scales),
            pair_features=pair_features,
        )
        return h_hat * w_hat, w_hat, h_hat
