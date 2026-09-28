"""Train and benchmark exact/amortized LK variants from a YAML suite."""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import time
from pathlib import Path

import torch

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required; install with `pip install pyyaml`.") from exc

from amortized_lk import AmortizedLK, sampled_covariance_ratio_loss, stochastic_probe_residual_loss
from evaluate_amortized_lk import exact_lk_graph
from lk_estimators import LKEstimator
from train_amortized_lk import stable_var_batch


def _median_seconds(fn, repeats: int, device: torch.device) -> float:
    with torch.no_grad():
        for _ in range(3):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)
    return statistics.median(times)


def _rank_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(), b.flatten()
    ra = torch.argsort(torch.argsort(a)).to(a.dtype)
    rb = torch.argsort(torch.argsort(b)).to(b.dtype)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float(((ra * rb).mean() / (ra.std(unbiased=False) * rb.std(unbiased=False)).clamp_min(1e-12)).item())


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction - target
    edge_abs = target.abs()
    strong = edge_abs >= edge_abs.quantile(0.5)
    sign_accuracy = (prediction[strong].sign() == target[strong].sign()).float().mean()
    threshold = edge_abs.quantile(0.75)
    truth = edge_abs > threshold
    guessed = prediction.abs() > threshold
    tp = (truth & guessed).sum().float()
    precision = tp / guessed.sum().clamp_min(1)
    recall = tp / truth.sum().clamp_min(1)
    support_f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "graph_mse": float(error.square().mean().item()),
        "graph_mae": float(error.abs().mean().item()),
        "magnitude_rank_corr": _rank_corr(prediction.abs(), target.abs()),
        "strong_sign_accuracy": float(sign_accuracy.item()),
        "support_f1_q75": float(support_f1.item()),
    }


def _example_batch(
    example: dict,
    batch: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Build one example family without computing any LK targets."""
    kind = example.get("generator", "stable_var")
    p, length = int(example["variables"]), int(example["time"])
    if kind == "stable_var":
        return stable_var_batch(
            batch, p, length, device=device, dtype=torch.float32,
            generator=generator,
            edge_probability=float(example.get("edge_probability", 0.2)),
        )
    if kind == "oscillator":
        if p != 2:
            raise ValueError("oscillator examples require exactly 2 variables")
        transition = torch.tensor(
            [[0.98, 0.08], [-0.12, 0.94]], device=device, dtype=torch.float32
        )
    elif kind == "repository6d":
        if p != 6:
            raise ValueError("repository6d examples require exactly 6 variables")
        transition = torch.tensor(
            [[0, 0, -0.6, 0, 0, 0],
             [-0.5, 0, 0, 0, 0, 0.8],
             [0, 0.7, 0, 0, 0, 0],
             [0, 0, 0, 0.7, 0.4, 0],
             [0, 0, 0, 0.2, 0, 0.7],
             [0, 0, 0, 0, 0, -0.5]],
            device=device,
            dtype=torch.float32,
        )
    else:
        raise ValueError(f"unknown example generator {kind!r}")
    innovations = torch.randn(batch, length, p, device=device, generator=generator)
    states = torch.zeros_like(innovations)
    states[:, 0] = innovations[:, 0]
    for t in range(1, length):
        states[:, t] = torch.matmul(transition, states[:, t - 1].unsqueeze(-1)).squeeze(-1) + innovations[:, t]
    return states.transpose(1, 2).contiguous()


def _train_variant(
    variant: dict,
    settings: dict,
    example: dict,
    *,
    seed: int,
    data_seed: int,
) -> tuple[AmortizedLK, float]:
    device = torch.device(settings["device"])
    torch.manual_seed(seed)
    model = AmortizedLK(
        width=variant.get("width", settings["training"]["width"]),
        projection_rank=variant.get("projection_rank", settings.get("projection_rank", 64)),
        n_step=example.get("n_step", 1),
        dt=example.get("dt", 1.0),
        h_method=(
            "projection" if variant.get("h_method", "projection") == "exact"
            else variant.get("h_method", "projection")
        ),
        encoder_type=variant.get("encoder_type", "tcn"),
        decoder_type=variant.get("decoder_type", "pair_mlp"),
        use_global_context=variant.get("use_global_context", True),
        low_rank=variant.get("low_rank", settings["training"].get("low_rank", 16)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=variant.get("learning_rate", settings["training"]["learning_rate"])
    )
    data_generator = torch.Generator(device=device).manual_seed(data_seed + 11)
    probe_generator = torch.Generator(device=device).manual_seed(seed + 29)
    training = settings["training"] | variant.get("training", {})
    start = time.perf_counter()
    for _ in range(example.get("training_steps", training["steps"])):
        x = _example_batch(
            example,
            example.get("training_batch_size", training["batch_size"]),
            device=device,
            generator=data_generator,
        )
        _, w_hat, h_hat = model(x, generator=probe_generator)
        loss = stochastic_probe_residual_loss(
            x,
            w_hat,
            q=training["probes"],
            ridge_scale=training["ridge_scale"],
            loss_mode=training["loss_mode"],
            probe_distribution=training["probe_distribution"],
            n_step=example.get("n_step", 1),
            dt=example.get("dt", 1.0),
            generator=probe_generator,
        )
        if variant.get("h_method", "projection") == "learned":
            pairs = torch.randint(
                example["variables"],
                (training["h_pairs"], 2),
                device=device,
                generator=probe_generator,
            )
            loss = loss + training["h_weight"] * sampled_covariance_ratio_loss(
                x,
                h_hat,
                pairs,
                n_step=example.get("n_step", 1),
                dt=example.get("dt", 1.0),
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if variant.get("h_method") == "exact":
        # H0 is evaluation-only; the coefficient model was optimized using the
        # same label-free residual as the H1/H2 variants.
        model.h_method = "exact"
    return model.eval(), time.perf_counter() - start


def run(config_path: Path) -> tuple[Path, Path]:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device(config.get("device", "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(int(config.get("num_threads", 1)))
    output_dir = Path(config.get("output_dir", "results/ablations"))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / "config.yaml")
    rows: list[dict] = []
    base_seed = int(config.get("seed", 0))

    for example_index, example in enumerate(config["examples"]):
        example_seed = base_seed + example_index * 100_000
        eval_generator = torch.Generator(device=device).manual_seed(example_seed + 7)
        evaluation = _example_batch(
            example,
            int(example.get("eval_examples", 8)),
            device=device,
            generator=eval_generator,
        )
        projection_generator = torch.Generator(device=device).manual_seed(example_seed + 9)
        projection_rank = int(config.get("projection_rank", 64))
        projection = torch.randn(
            projection_rank,
            example["time"] - int(example.get("n_step", 1)),
            device=device,
            generator=projection_generator,
        ) / projection_rank**0.5

        fitted = []
        for variant_index, variant in enumerate(config["variants"]):
            if variant.get("mode") in {"exact", "exact_graph"}:
                fitted.append((variant, None, 0.0))
            elif variant.get("mode") == "amortized":
                model, train_seconds = _train_variant(
                    variant,
                    config,
                    example,
                    seed=example_seed + variant_index * 1000,
                    data_seed=example_seed,
                )
                fitted.append((variant, model, train_seconds))
            else:
                raise ValueError(f"unknown mode in variant {variant['name']!r}")
            print(f"prepared {example['name']} / {variant['name']}")

        # Exact LK is evaluated only after all learned variants have finished training.
        exact_estimator = LKEstimator(
            mode="exact",
            n_step=int(example.get("n_step", 1)),
            dt=float(example.get("dt", 1.0)),
        ).to(device).eval()
        with torch.no_grad():
            reference = exact_estimator(evaluation).graph
            graph_only_reference = exact_lk_graph(
                evaluation,
                n_step=int(example.get("n_step", 1)),
                dt=float(example.get("dt", 1.0)),
            )[0]
        exact_seconds = _median_seconds(
            lambda: exact_estimator(evaluation), int(config.get("timing_repeats", 20)), device
        )
        graph_only_seconds = _median_seconds(
            lambda: exact_lk_graph(
                evaluation,
                n_step=int(example.get("n_step", 1)),
                dt=float(example.get("dt", 1.0)),
            ),
            int(config.get("timing_repeats", 20)), device,
        )

        for variant, model, train_seconds in fitted:
            if variant.get("mode") == "exact":
                prediction = reference
                inference_seconds = exact_seconds
            elif variant.get("mode") == "exact_graph":
                prediction = graph_only_reference
                inference_seconds = graph_only_seconds
            else:
                assert model is not None
                fixed_projection = projection if variant.get("h_method", "projection") == "projection" else None
                estimator = LKEstimator(
                    mode="amortized", amortized_model=model
                ).to(device).eval()
                with torch.no_grad():
                    prediction = estimator(evaluation, projection=fixed_projection).graph
                inference_seconds = _median_seconds(
                    lambda: estimator(evaluation, projection=fixed_projection),
                    int(config.get("timing_repeats", 20)), device,
                )
            metrics = _metrics(prediction, reference)
            row = {
                "example": example["name"],
                "variables": example["variables"],
                "time": example["time"],
                "variant": variant["name"],
                "mode": variant["mode"],
                "train_seconds": train_seconds,
                "inference_seconds": inference_seconds,
                "speedup_vs_exact": exact_seconds / max(inference_seconds, 1e-12),
                **metrics,
            }
            rows.append(row)
            print(
                f"measured {row['example']} / {row['variant']}: "
                f"{row['inference_seconds']:.6g}s, mse={row['graph_mse']:.6g}"
            )

    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = output_dir / "results.md"
    lines = [
        "# CPU LK ablation results",
        "",
        f"Config: `config.yaml`; PyTorch {torch.__version__}; device `{device}`.",
        "The `exact_lk` timer includes the repository's variance calculation; `exact_graph_only` isolates graph estimation. Learned timing reports graph inference and does not produce variance.",
        "",
        "| Example | Variant | Train (s) | Inference (ms) | Speedup | Graph MSE | Magnitude rank corr. | Strong sign acc. |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['example']} (p={row['variables']}, T={row['time']}) "
            f"| {row['variant']} | {row['train_seconds']:.2f} "
            f"| {1000*row['inference_seconds']:.3f} "
            f"| {row['speedup_vs_exact']:.2f}x | {row['graph_mse']:.6g} "
            f"| {row['magnitude_rank_corr']:.4f} | {row['strong_sign_accuracy']:.4f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML benchmark configuration")
    args = parser.parse_args()
    csv_path, markdown_path = run(args.config)
    print(f"saved {csv_path}")
    print(f"saved {markdown_path}")


if __name__ == "__main__":
    main()
