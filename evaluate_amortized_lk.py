"""Evaluate a trained amortized LK model against the exact LK oracle.

This script is intentionally separate from training. The exact linear solve
appears only here, after loading a checkpoint, for evaluation and diagnostics.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from amortized_lk import exact_covariance_ratio
from lk_estimators import load_amortized_checkpoint
from train_amortized_lk import stable_var_batch


def exact_lk_graph(x: torch.Tensor, *, n_step: int = 1, dt: float = 1.0):
    """Exact OLS LK graph used solely as a post-training evaluation oracle."""
    z = x[..., :-n_step]
    y = (x[..., n_step:] - x[..., :-n_step]) / (n_step * dt)
    z = z - z.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    m = z.shape[-1]
    covariance = z @ z.transpose(1, 2) / (m - 1)
    cross_covariance = z @ y.transpose(1, 2) / (m - 1)
    try:
        coefficients = torch.linalg.solve(covariance, cross_covariance)
    except RuntimeError:
        coefficients = torch.linalg.pinv(covariance) @ cross_covariance
    ratio = exact_covariance_ratio(x, n_step=n_step, dt=dt)
    return ratio * coefficients, coefficients, ratio


def rank_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of average ranks, computed without SciPy."""
    a = a.flatten()
    b = b.flatten()
    ra = torch.argsort(torch.argsort(a)).to(a.dtype)
    rb = torch.argsort(torch.argsort(b)).to(b.dtype)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return (ra * rb).mean() / (
        ra.square().mean().sqrt() * rb.square().mean().sqrt()
    ).clamp_min(torch.finfo(a.dtype).eps)


def median_runtime(function, device: torch.device, repeats: int) -> float:
    for _ in range(3):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--variables", type=int, default=6)
    parser.add_argument("--time", type=int, default=512)
    parser.add_argument("--projection-rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timing-repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_amortized_checkpoint(
        str(args.checkpoint), map_location=device,
        projection_rank=args.projection_rank,
    ).to(device)
    model.eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    windows = stable_var_batch(
        args.examples,
        args.variables,
        args.time,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    with torch.no_grad():
        predicted, w_hat, h_hat = model(windows, generator=generator)
        exact, w_exact, h_exact = exact_lk_graph(windows)
        amortized_seconds = median_runtime(
            lambda: model(windows, generator=generator), device, args.timing_repeats
        )
        exact_seconds = median_runtime(
            lambda: exact_lk_graph(windows), device, args.timing_repeats
        )

    error = predicted - exact
    edge_margin = exact.abs()
    strong = edge_margin >= edge_margin.quantile(0.5)
    sign_accuracy = (predicted[strong].sign() == exact[strong].sign()).float().mean()
    max_error = error.abs().max()
    sign_certified = edge_margin > max_error
    sign_margin_accuracy = (
        (predicted[sign_certified].sign() == exact[sign_certified].sign()).float().mean()
        if sign_certified.any()
        else error.new_tensor(float("nan"))
    )
    flat_exact = exact.abs().flatten()
    flat_predicted = predicted.abs().flatten()
    pair_count = min(100_000, flat_exact.numel() ** 2)
    pair_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    pair_indices = torch.randint(
        flat_exact.numel(), (2, pair_count), device=device, generator=pair_generator
    )
    exact_pair_gap = flat_exact[pair_indices[0]] - flat_exact[pair_indices[1]]
    pair_gap = exact_pair_gap.abs()
    predicted_gap = flat_predicted[pair_indices[0]] - flat_predicted[pair_indices[1]]
    rank_certified = pair_gap > (2 * max_error)
    rank_margin_accuracy = (
        (exact_pair_gap[rank_certified] * predicted_gap[rank_certified] > 0)
        .float().mean()
        if rank_certified.any()
        else error.new_tensor(float("nan"))
    )
    threshold = edge_margin.quantile(0.75)
    exact_support = edge_margin > threshold
    predicted_support = predicted.abs() > threshold
    true_positive = (exact_support & predicted_support).sum().float()
    precision = true_positive / predicted_support.sum().clamp_min(1)
    recall = true_positive / exact_support.sum().clamp_min(1)
    support_f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    h0_graph = h_exact * w_hat
    print(f"graph_mse={error.square().mean().item():.6g}")
    print(f"graph_mae={error.abs().mean().item():.6g}")
    print(f"magnitude_rank_correlation={rank_correlation(predicted.abs(), exact.abs()).item():.6g}")
    print(f"strong_edge_sign_accuracy={sign_accuracy.item():.6g}")
    print(f"max_graph_error={max_error.item():.6g}")
    print(f"sign_margin_fraction={sign_certified.float().mean().item():.6g}")
    print(f"sign_margin_accuracy={sign_margin_accuracy.item():.6g}")
    print(f"rank_margin_accuracy={rank_margin_accuracy.item():.6g}")
    print(f"threshold_support_f1={support_f1.item():.6g}")
    print(f"coefficient_mse={(w_hat - w_exact).square().mean().item():.6g}")
    print(f"covariance_ratio_mse={(h_hat - h_exact).square().mean().item():.6g}")
    print(f"learned_graph_with_exact_H_mse={(h0_graph - exact).square().mean().item():.6g}")
    print(f"amortized_seconds={amortized_seconds:.6g}")
    print(f"exact_seconds={exact_seconds:.6g}")


if __name__ == "__main__":
    main()
