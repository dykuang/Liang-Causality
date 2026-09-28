"""Run compact Stage-1 math/probe/ridge/H-sketch experiments."""

from __future__ import annotations

import argparse

import torch

from amortized_lk import (
    exact_covariance_ratio,
    random_projection_covariance_ratio,
    stochastic_probe_residual_loss,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes-per-setting", type=int, default=128)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    dtype = torch.float64
    p, m = 8, 192

    # Correlated channels with a broad but controlled covariance spectrum.
    raw = torch.randn(1, p, m + 1, device=device, dtype=dtype, generator=generator)
    scales = torch.logspace(0, -2, p, device=device, dtype=dtype)
    mixing = torch.randn(p, p, device=device, dtype=dtype, generator=generator)
    orthogonal, _ = torch.linalg.qr(mixing)
    transform = orthogonal @ torch.diag(scales) @ orthogonal.T
    x = transform @ raw[0]
    x = x.unsqueeze(0)
    w = torch.randn(1, p, p, device=device, dtype=dtype, generator=generator)

    z = x[..., :-1]
    y = x[..., 1:] - x[..., :-1]
    z = z - z.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    covariance = z @ z.transpose(1, 2) / (m - 1)
    cross = z @ y.transpose(1, 2) / (m - 1)
    lam = 1e-3 * torch.diagonal(covariance, dim1=-2, dim2=-1).mean()
    residual = (covariance + lam * torch.eye(p, device=device, dtype=dtype)) @ w - cross
    full_loss = residual.square().sum().item()
    print(f"full_ridge_residual={full_loss:.8g}")

    print("probe sweep: q, distribution, mean, std, relative_mean_error")
    for distribution in ("rademacher", "gaussian", "orthogonal"):
        for q in (1, 2, 4, 8):
            estimates = []
            for _ in range(args.probes_per_setting):
                estimates.append(
                    stochastic_probe_residual_loss(
                        x,
                        w,
                        q=q,
                        ridge=float(lam),
                        probe_distribution=distribution,
                        generator=generator,
                    ).item()
                )
            sample = torch.tensor(estimates, dtype=dtype)
            relative_error = abs(sample.mean().item() - full_loss) / max(full_loss, 1e-12)
            print(
                f"{q},{distribution},{sample.mean().item():.8g},"
                f"{sample.std(unbiased=True).item():.8g},{relative_error:.8g}"
            )

    eigenvalues = torch.linalg.eigvalsh(covariance[0])
    min_eigenvalue = eigenvalues.min()
    max_eigenvalue = eigenvalues.max()
    w_unregularized = torch.linalg.solve(covariance, cross)
    print(
        f"ridge sweep: lambda, coefficient_bias_fro, residual_fro, "
        f"condition_number={max_eigenvalue.item() / min_eigenvalue.item():.8g}"
    )
    for multiplier in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
        ridge = multiplier * covariance.diagonal(dim1=-2, dim2=-1).mean()
        solution = torch.linalg.solve(
            covariance + ridge * torch.eye(p, device=device, dtype=dtype), cross
        )
        equation_residual = (covariance + ridge * torch.eye(p, device=device, dtype=dtype)) @ solution - cross
        print(
            f"{ridge.item():.8g},{(solution - w_unregularized).norm().item():.8g},"
            f"{equation_residual.norm().item():.8g}"
        )

    h_exact = exact_covariance_ratio(x)
    print("H1 rank sweep: rank, ratio_mse")
    for rank in (4, 8, 16, 32):
        estimates = []
        for _ in range(args.probes_per_setting):
            h_est = random_projection_covariance_ratio(
                x, rank=rank, generator=generator
            )
            estimates.append((h_est - h_exact).square().mean().item())
        values = torch.tensor(estimates, dtype=dtype)
        print(f"{rank},{values.mean().item():.8g} +/- {values.std(unbiased=True).item():.8g}")


if __name__ == "__main__":
    main()
