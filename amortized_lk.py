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


def _state_and_derivative(
    x: Tensor, n_step: int, dt: float
) -> tuple[Tensor, Tensor]:
    """Return centered state and finite-difference matrices as ``[B,p,m]``."""
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, variables, time]")
    if n_step < 1 or x.shape[-1] <= n_step + 1:
        raise ValueError("time dimension must be greater than n_step + 1")
    if dt <= 0:
        raise ValueError("dt must be positive")

    z = x[..., :-n_step]
    y = (x[..., n_step:] - x[..., :-n_step]) / (n_step * dt)
    z = z - z.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    return z, y


def stochastic_probe_residual_loss(
    x: Tensor,
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
    """
    if q < 1:
        raise ValueError("q must be at least 1")
    z, y = _state_and_derivative(x, n_step, dt)
    batch, variables, m = z.shape
    if w_hat.shape != (batch, variables, variables):
        raise ValueError("w_hat must have shape [batch, variables, variables]")

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
            batch, variables, q, device=x.device, dtype=x.dtype
        ).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
    elif probe_distribution == "gaussian":
        probes = torch.randn(
            batch, variables, q, device=x.device, dtype=x.dtype,
            generator=generator,
        )
    elif probe_distribution == "orthogonal":
        if q > variables:
            raise ValueError("orthogonal probes require q <= number of variables")
        gaussian = torch.randn(
            batch, variables, q, device=x.device, dtype=x.dtype,
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
        scale = dv.square().sum(dim=1).clamp_min(torch.finfo(x.dtype).eps)
        probe_loss = probe_loss / scale
    elif mode == "jacobi":
        diagonal = z.square().sum(dim=-1) / (m - 1)
        preconditioner = (diagonal + lam[:, None]).clamp_min(
            torch.finfo(x.dtype).eps
        ).rsqrt()
        probe_loss = (residual_samples * preconditioner.unsqueeze(-1)).square().sum(dim=1)
    elif mode != "plain":
        raise ValueError("loss_mode must be plain, normalized, or jacobi")
    return probe_loss.mean()


def random_projection_covariance_ratio(
    x: Tensor,
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
    """
    if rank < 1:
        raise ValueError("rank must be at least 1")
    z, _ = _state_and_derivative(x, n_step, dt)
    batch, variables, m = z.shape
    if projection is None:
        projection = torch.randn(
            rank, m, dtype=x.dtype, device=x.device, generator=generator
        ) / math.sqrt(rank)
    else:
        if projection.shape != (rank, m):
            raise ValueError("projection must have shape [rank, time-n_step]")
        projection = projection.to(device=x.device, dtype=x.dtype)
    sketch = z @ projection.transpose(0, 1)
    gram = sketch @ sketch.transpose(1, 2)
    diagonal = torch.diagonal(gram, dim1=-2, dim2=-1)
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    return gram / diagonal.unsqueeze(-2).clamp_min(eps)


def exact_covariance_ratio(
    x: Tensor, *, n_step: int = 1, dt: float = 1.0, eps: Optional[float] = None
) -> Tensor:
    """Compute the exact sample ``H`` ratio for evaluation-only H0 baselines."""
    z, _ = _state_and_derivative(x, n_step, dt)
    gram = z @ z.transpose(1, 2)
    diagonal = torch.diagonal(gram, dim1=-2, dim2=-1)
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    return gram / diagonal.unsqueeze(-2).clamp_min(eps)


def sampled_covariance_ratio_loss(
    x: Tensor, h_hat: Tensor, pairs: Tensor, *, n_step: int = 1,
    dt: float = 1.0, eps: Optional[float] = None,
) -> Tensor:
    """Supervise selected H[i,j] ratios directly, without an LK label or full H."""
    z, _ = _state_and_derivative(x, n_step, dt)
    batch, variables, _ = z.shape
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [num_pairs, 2]")
    if h_hat.shape != (batch, variables, variables):
        raise ValueError("h_hat must have shape [batch, variables, variables]")
    pairs = pairs.to(device=x.device, dtype=torch.long)
    i, j = pairs.unbind(dim=1)
    numerator = (z[:, i] * z[:, j]).sum(dim=-1)
    denominator = z[:, j].square().sum(dim=-1)
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    target = numerator / denominator.clamp_min(eps)
    prediction = h_hat[:, i, j]
    return (prediction - target).square().mean()


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
    """Permutation-equivariant direct predictor of an approximate LK graph.

    The model predicts ``W_hat`` using shared per-variable temporal features,
    optional DeepSets-style global context, and a configurable shared decoder. It obtains
    ``H_hat`` through a randomized temporal projection (H1 in the design note)
    and returns ``T_hat = H_hat * W_hat``. ``h_method="projection"`` selects
    H1, ``"learned"`` selects H2, and ``"exact"`` computes the exact sample H
    ratio as an H0 evaluation baseline. H0 does not solve for LK coefficients.
    No dense LK solve is used at inference or by
    :func:`stochastic_probe_residual_loss` during training.
    The encoder sees centered, RMS-scaled, sign-canonicalized channels, making
    its representation invariant to channel offsets and nonzero changes of
    measurement units. Coefficients are converted back to input units.
    """

    def __init__(
        self,
        width: int = 32,
        projection_rank: int = 64,
        n_step: int = 1,
        dt: float = 1.0,
        h_method: str = "projection",
        encoder_type: str = "tcn",
        decoder_type: str = "pair_mlp",
        use_global_context: bool = True,
        low_rank: int = 16,
    ):
        super().__init__()
        if width < 1:
            raise ValueError("width must be at least 1")
        if projection_rank < 1:
            raise ValueError("projection_rank must be at least 1")
        self.encoder = _make_encoder(encoder_type, width)
        self.encoder_type = encoder_type
        decoder_types = {"pair_mlp", "bilinear", "low_rank", "bilinear_residual"}
        if decoder_type not in decoder_types:
            raise ValueError(f"unknown decoder_type {decoder_type!r}; choose {sorted(decoder_types)}")
        if low_rank < 1:
            raise ValueError("low_rank must be at least 1")
        self.decoder_type = decoder_type
        self.use_global_context = use_global_context
        self.low_rank = low_rank
        self.global_phi = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.global_rho = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.edge_decoder = nn.Sequential(
            nn.Linear(5 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        ) if decoder_type in {"pair_mlp", "bilinear_residual"} else None
        self.bilinear_decoder = BilinearEdgeDecoder(width) if decoder_type in {
            "bilinear", "bilinear_residual"
        } else None
        self.low_rank_decoder = LowRankEdgeDecoder(width, low_rank) if decoder_type == "low_rank" else None
        self.projection_rank = projection_rank
        self.n_step = n_step
        self.dt = dt
        if h_method not in {"projection", "learned", "exact"}:
            raise ValueError("h_method must be projection, learned, or exact")
        self.h_method = h_method
        self.h_decoder = nn.Sequential(
            nn.Linear(5 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        ) if h_method == "learned" else None

    def forward(
        self,
        x: Tensor,
        *,
        projection: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, variables, time]")
        batch, variables, _ = x.shape
        centered = x - x.mean(dim=-1, keepdim=True)
        rms = centered.square().mean(dim=-1, keepdim=True).sqrt()
        rms = rms.clamp_min(torch.finfo(x.dtype).eps)
        anchor = centered.gather(-1, centered.abs().argmax(dim=-1, keepdim=True))
        orientation = torch.where(anchor < 0, -torch.ones_like(anchor), torch.ones_like(anchor))
        signed_scales = rms * orientation
        h = self.encoder(centered / signed_scales)
        if self.use_global_context:
            global_context = self.global_rho(self.global_phi(h).mean(dim=1))
        else:
            global_context = torch.zeros_like(h.mean(dim=1))

        needs_pair_features = (
            self.decoder_type in {"pair_mlp", "bilinear_residual"}
            or self.h_method == "learned"
        )
        source = h.unsqueeze(2).expand(-1, -1, variables, -1)
        target = h.unsqueeze(1).expand(-1, variables, -1, -1)
        pair_features = None
        if needs_pair_features:
            context = global_context[:, None, None, :].expand(-1, variables, variables, -1)
            pair_features = torch.cat(
                (source, target, source * target, source - target, context), dim=-1
            )
        if self.decoder_type == "pair_mlp":
            assert pair_features is not None
            w_normalized = self.edge_decoder(pair_features).squeeze(-1)
        elif self.decoder_type == "bilinear":
            w_normalized = self.bilinear_decoder(source, target, global_context)
        elif self.decoder_type == "low_rank":
            w_normalized = self.low_rank_decoder(h, global_context)
        else:
            assert pair_features is not None
            w_normalized = (
                self.bilinear_decoder(source, target, global_context)
                + self.edge_decoder(pair_features).squeeze(-1)
            )
        # If x_i is divided by signed s_i before encoding, its coefficient
        # matrix is W_normalized[i,j] = W_physical[i,j] * s_i / s_j.
        scales = signed_scales.squeeze(-1)
        w_hat = w_normalized * scales.unsqueeze(1) / scales.unsqueeze(2)
        if self.h_method == "projection":
            h_hat = random_projection_covariance_ratio(
                x,
                rank=self.projection_rank,
                n_step=self.n_step,
                dt=self.dt,
                projection=projection,
                generator=generator,
            )
        elif self.h_method == "learned":
            assert pair_features is not None
            h_hat = self.h_decoder(pair_features).squeeze(-1)
        else:
            if self.training:
                raise RuntimeError("h_method='exact' is evaluation-only; call eval() first")
            h_hat = exact_covariance_ratio(
                x, n_step=self.n_step, dt=self.dt
            )
        t_hat = h_hat * w_hat
        return t_hat, w_hat, h_hat
