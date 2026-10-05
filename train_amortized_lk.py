"""Train the amortized LK operator with residual and optional sampled-H losses.

This entry point deliberately has no dependency on the exact LK estimator.
The built-in data source generates stable VAR windows without computing LK
coefficients or causal-matrix labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from amortized_lk import (
    AmortizedLK,
    sampled_covariance_ratio_loss,
    sampled_covariance_ratio_pair_loss,
    stochastic_probe_residual_loss,
)


def stable_var_batch(
    batch: int,
    variables: int,
    time: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    edge_probability: float = 0.2,
) -> torch.Tensor:
    """Generate stable signed VAR windows; this produces no LK targets."""
    coefficients = torch.randn(
        batch, variables, variables, device=device, dtype=dtype,
        generator=generator,
    )
    mask = torch.rand(
        batch, variables, variables, device=device, dtype=dtype,
        generator=generator,
    ) < edge_probability
    coefficients = coefficients * mask
    row_norm = coefficients.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    coefficients = coefficients * (0.8 / row_norm)
    innovations = torch.randn(
        batch, time, variables, device=device, dtype=dtype,
        generator=generator,
    )
    states = torch.zeros_like(innovations)
    states[:, 0] = innovations[:, 0]
    for t in range(1, time):
        states[:, t] = torch.bmm(
            coefficients, states[:, t - 1].unsqueeze(-1)
        ).squeeze(-1) + innovations[:, t]
    return states.transpose(1, 2).contiguous()


def train_amortized_step(
    model: AmortizedLK,
    optimizer: torch.optim.Optimizer,
    windows: torch.Tensor,
    *,
    probes: int,
    ridge_scale: float,
    loss_mode: str,
    probe_distribution: str,
    generator: torch.Generator,
    n_step: int = 1,
    dt: float = 1.0,
    h_weight: float = 0.0,
    h_pairs: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one coefficient-residual update and optional sampled-H update."""
    has_sampled_h_objective = (
        h_weight != 0
        and h_pairs > 0
        and model.h_method in {"learned_pair", "learned_low_rank"}
    )
    predict_dense_h = has_sampled_h_objective and model.h_method == "learned_pair"
    _, coefficients, h_hat = model(
        windows, generator=generator, compute_graph=predict_dense_h
    )
    equation_loss = stochastic_probe_residual_loss(
        windows,
        coefficients,
        q=probes,
        ridge_scale=ridge_scale,
        loss_mode=loss_mode,
        probe_distribution=probe_distribution,
        n_step=n_step,
        dt=dt,
        generator=generator,
    )
    h_loss = equation_loss.new_zeros(())
    if has_sampled_h_objective:
        pairs = torch.randint(
            windows.shape[1],
            (h_pairs, 2),
            device=windows.device,
            generator=generator,
        )
        if model.h_method == "learned_low_rank":
            predictions = model.score_h_pairs(windows, pairs)
            h_loss = sampled_covariance_ratio_pair_loss(
                windows, predictions, pairs, n_step=n_step, dt=dt
            )
        else:
            if h_hat is None:
                raise RuntimeError("learned_pair training requires predicted H")
            h_loss = sampled_covariance_ratio_loss(
                windows, h_hat, pairs, n_step=n_step, dt=dt
            )
    total_loss = equation_loss + h_weight * h_loss
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    optimizer.step()
    return equation_loss.detach(), h_loss.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--variables", type=int, default=6)
    parser.add_argument("--time", type=int, default=256)
    parser.add_argument("--variables-min", type=int)
    parser.add_argument("--variables-max", type=int)
    parser.add_argument("--time-min", type=int)
    parser.add_argument("--time-max", type=int)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--encoder-type", choices=("tcn", "downsample_mlp", "spectral", "attention"), default="tcn")
    parser.add_argument("--decoder-type", choices=("pair_mlp", "bilinear", "low_rank", "bilinear_residual"), default="pair_mlp")
    parser.add_argument("--no-global-context", action="store_true")
    parser.add_argument("--low-rank", type=int, default=16)
    parser.add_argument("--projection-rank", type=int, default=64)
    parser.add_argument("--probes", type=int, default=4)
    parser.add_argument("--ridge-scale", type=float, default=1e-3)
    parser.add_argument("--loss-mode", choices=("plain", "normalized", "jacobi"), default="plain")
    parser.add_argument("--probe-distribution", choices=("rademacher", "gaussian", "orthogonal"), default="rademacher")
    parser.add_argument(
        "--h-method", choices=("projection", "learned_pair", "learned"),
        default="projection",
    )
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--h-weight", type=float, default=1.0)
    parser.add_argument("--h-pairs", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("amortized_lk.pt"))
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    model = AmortizedLK(
        width=args.width,
        projection_rank=args.projection_rank,
        h_method=args.h_method,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        use_global_context=not args.no_global_context,
        low_rank=args.low_rank,
        n_step=args.n_step,
        dt=args.dt,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for step in range(1, args.steps + 1):
        variables = args.variables
        if args.variables_min is not None or args.variables_max is not None:
            lower = args.variables_min if args.variables_min is not None else args.variables
            upper = args.variables_max if args.variables_max is not None else args.variables
            if lower < 1 or upper < lower:
                raise ValueError("variable range must satisfy 1 <= min <= max")
            variables = int(torch.randint(lower, upper + 1, (), generator=generator, device=device).item())
        time = args.time
        if args.time_min is not None or args.time_max is not None:
            lower = args.time_min if args.time_min is not None else args.time
            upper = args.time_max if args.time_max is not None else args.time
            if lower < 3 or upper < lower:
                raise ValueError("time range must satisfy 3 <= min <= max")
            time = int(torch.randint(lower, upper + 1, (), generator=generator, device=device).item())
        windows = stable_var_batch(
            args.batch_size,
            variables,
            time,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        equation_loss, h_loss = train_amortized_step(
            model,
            optimizer,
            windows,
            probes=args.probes,
            ridge_scale=args.ridge_scale,
            loss_mode=args.loss_mode,
            probe_distribution=args.probe_distribution,
            generator=generator,
            n_step=args.n_step,
            dt=args.dt,
            h_weight=args.h_weight,
            h_pairs=args.h_pairs,
        )

        if step == 1 or step % 100 == 0 or step == args.steps:
            print(
                f"step={step} probe_residual={equation_loss.item():.6g} "
                f"sampled_h={h_loss.item():.6g}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(args) | {
                "output": str(args.output), "h_method": model.h_method,
            },
        },
        args.output,
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
