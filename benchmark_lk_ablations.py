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

from amortized_lk import AmortizedLK, exact_covariance_ratio
from evaluate_amortized_lk import exact_lk_graph
from lk_estimators import LKEstimator
from train_amortized_lk import stable_var_batch, train_amortized_step


def _timing(fn, repeats: int, warmup: int, device: torch.device) -> tuple[float, float]:
    """Return CUDA-event (or wall) and synchronized wall seconds."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            event_times, wall_times = [], []
            for _ in range(repeats):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record(); fn(); end.record(); end.synchronize()
                event_times.append(start.elapsed_time(end) / 1000)
                wall_times.append(time.perf_counter() - wall_start)
            return statistics.median(event_times), statistics.median(wall_times)
        times = []
        for _ in range(repeats):
            start = time.perf_counter(); fn(); times.append(time.perf_counter() - start)
    return statistics.median(times), statistics.median(times)


def _exact_graph_solve_ex(x: torch.Tensor, n_step: int = 1, dt: float = 1.0) -> torch.Tensor:
    z = x[..., :-n_step]
    y = (x[..., n_step:] - x[..., :-n_step]) / (n_step * dt)
    z, y = z - z.mean(-1, keepdim=True), y - y.mean(-1, keepdim=True)
    denom = z.shape[-1] - 1
    c, d = z @ z.transpose(1, 2) / denom, z @ y.transpose(1, 2) / denom
    w, info = torch.linalg.solve_ex(c, d, check_errors=False)
    if torch.any(info != 0):
        w = torch.linalg.solve(c, d)
    h = c / torch.diagonal(c, dim1=-2, dim2=-1).unsqueeze(-2).clamp_min(torch.finfo(x.dtype).eps)
    return h * w


def _rank_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    ra = torch.argsort(torch.argsort(a)).to(a.dtype)
    rb = torch.argsort(torch.argsort(b)).to(b.dtype)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float(((ra * rb).mean() / (ra.std(unbiased=False) * rb.std(unbiased=False)).clamp_min(1e-12)).item())


def _h_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Measure H error and off-diagonal magnitude ordering."""
    prediction, target = prediction.float(), target.float()
    variables = prediction.shape[-1]
    off_diagonal = ~torch.eye(variables, dtype=torch.bool, device=prediction.device)
    return {
        "h_mse": float((prediction - target).square().mean().item()),
        "magnitude_rank_corr": _rank_corr(
            prediction[..., off_diagonal].abs(), target[..., off_diagonal].abs()
        ),
    }


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction, target = prediction.float(), target.float()
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
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build one example family without computing any LK targets."""
    kind = example.get("generator", "stable_var")
    p, length = int(example["variables"]), int(example["time"])
    if kind == "stable_var":
        return stable_var_batch(
            batch, p, length, device=device, dtype=dtype,
            generator=generator,
            edge_probability=float(example.get("edge_probability", 0.2)),
        )
    if kind == "oscillator":
        if p != 2:
            raise ValueError("oscillator examples require exactly 2 variables")
        transition = torch.tensor(
            [[0.98, 0.08], [-0.12, 0.94]], device=device, dtype=dtype
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
            dtype=dtype,
        )
    else:
        raise ValueError(f"unknown example generator {kind!r}")
    innovations = torch.randn(
        batch, length, p, device=device, dtype=dtype, generator=generator
    )
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
    device = torch.device(settings["runtime"]["device"])
    dtype = settings["dtype"]
    torch.manual_seed(seed)
    model = AmortizedLK(
        width=variant.get("width", settings["training"]["width"]),
        projection_rank=variant.get("projection_rank", settings.get("projection_rank", 64)),
        n_step=example.get("n_step", 1),
        dt=example.get("dt", 1.0),
        h_method=variant.get("h_method", "projection"),
        encoder_type=variant.get("encoder_type", "tcn"),
        decoder_type=variant.get("decoder_type", "pair_mlp"),
        use_global_context=variant.get("use_global_context", True),
        low_rank=variant.get("low_rank", settings["training"].get("low_rank", 16)),
        h_rank=variant.get("h_rank", settings["training"].get("h_rank", 8)),
        h_factor_hidden=variant.get("h_factor_hidden"),
        h_structure=variant.get("h_structure", "correlation_psd"),
    ).to(device=device, dtype=dtype)
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
            generator=data_generator, dtype=dtype,
        )
        train_amortized_step(
            model,
            optimizer,
            x,
            probes=training["probes"],
            ridge_scale=training["ridge_scale"],
            loss_mode=training["loss_mode"],
            probe_distribution=training["probe_distribution"],
            generator=probe_generator,
            n_step=example.get("n_step", 1),
            dt=example.get("dt", 1.0),
            h_weight=training.get("h_weight", 0.0),
            h_pairs=training.get("h_pairs", 0),
        )
    return model.eval(), time.perf_counter() - start


def _canonicalize_h_method_aliases(config: dict) -> bool:
    """Rewrite accepted legacy H names to their canonical serialized names."""
    changed = False
    for variant in config.get("variants", []):
        if variant.get("h_method") == "learned":
            variant["h_method"] = "learned_pair"
            changed = True
    return changed


def validate_config(config: dict) -> dict:
    _canonicalize_h_method_aliases(config)
    runtime = dict(config.get("runtime", {}))
    runtime.setdefault("device", config.get("device", "cpu"))
    runtime.setdefault("dtype", "float32")
    runtime.setdefault("compile", False)
    runtime.setdefault("compile_mode", "reduce-overhead")
    runtime.setdefault("warmup_repeats", 3 if runtime["device"] == "cpu" else 10)
    runtime.setdefault("timing_repeats", config.get("timing_repeats", 20))
    runtime.setdefault("synchronize_cuda", True)
    if runtime["dtype"] not in {"float32", "bfloat16", "float16"}:
        raise ValueError("runtime.dtype must be float32, bfloat16, or float16")
    if runtime["device"] == "cpu" and runtime["dtype"] != "float32":
        raise ValueError("mixed precision is supported only on CUDA")
    if runtime["compile"] and not hasattr(torch, "compile"):
        raise RuntimeError("runtime.compile=true but torch.compile is unavailable")
    config["runtime"] = runtime
    config["dtype"] = getattr(torch, runtime["dtype"])
    for variant in config["variants"]:
        if variant.get("h_method") == "projection" and int(variant.get("projection_rank", config.get("projection_rank", 64))) < 1:
            raise ValueError("projection_rank must be positive")
        if variant.get("h_method") == "learned_low_rank" and int(variant.get("h_rank", config["training"].get("h_rank", 8))) < 1:
            raise ValueError("h_rank must be positive")
        if variant.get("decoder_type") == "low_rank" and int(variant.get("low_rank", config["training"].get("low_rank", 16))) < 1:
            raise ValueError("low_rank must be positive")
    return config


def result_precision_metadata(
    requested_dtype: str, graph_output: torch.Tensor
) -> dict[str, str]:
    """Describe requested precision and the dtype of the measured graph output."""
    return {
        "requested_dtype": requested_dtype,
        "effective_dtype": str(graph_output.dtype).removeprefix("torch."),
    }


def run(config_path: Path) -> tuple[Path, Path]:
    config_path = config_path.resolve()
    config_text = config_path.read_text(encoding="utf-8")
    saved_config = yaml.safe_load(config_text)
    has_canonicalized_aliases = _canonicalize_h_method_aliases(saved_config)
    config = validate_config(yaml.safe_load(config_text))
    runtime, dtype = config["runtime"], config["dtype"]
    device = torch.device(runtime["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cpu": torch.set_num_threads(int(config.get("num_threads", 1)))
    output_dir = Path(config.get("output_dir", "results/ablations"))
    output_dir = (Path.cwd() / output_dir if not output_dir.is_absolute() else output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_config_path = output_dir / "config.yaml"
    if copied_config_path != config_path:
        if has_canonicalized_aliases:
            copied_config_path.write_text(
                yaml.safe_dump(saved_config, sort_keys=False), encoding="utf-8"
            )
        else:
            shutil.copyfile(config_path, copied_config_path)
    rows, base_seed = [], int(config.get("seed", 0))
    for example_index, example in enumerate(config["examples"]):
        seed = base_seed + example_index * 100_000
        evaluation = _example_batch(example, int(example.get("eval_examples", 8)), device=device, generator=torch.Generator(device=device).manual_seed(seed + 7), dtype=dtype)
        fitted = []
        for variant in config["variants"]:
            if variant.get("mode") in {"exact", "exact_graph", "exact_graph_solve_ex"}: fitted.append((variant, None, 0.0))
            elif variant.get("mode") == "amortized": fitted.append((variant, *_train_variant(variant, config, example, seed=seed, data_seed=seed)))
            else: raise ValueError(f"unknown mode {variant.get('mode')!r}")
        n_step, dt = int(example.get("n_step", 1)), float(example.get("dt", 1.0))
        exact_estimator = LKEstimator(mode="exact", n_step=n_step, dt=dt).to(device=device, dtype=torch.float32).eval()
        exact_input = evaluation.float() if dtype != torch.float32 else evaluation
        with torch.no_grad():
            exact_reference = exact_estimator(exact_input).graph
            graph_only_reference = exact_lk_graph(exact_input, n_step=n_step, dt=dt)[0]
            reference = exact_reference
            graph_reference = graph_only_reference
            h_reference = exact_covariance_ratio(evaluation, n_step=n_step, dt=dt)
        exact_time = _timing(lambda: exact_estimator(exact_input), runtime["timing_repeats"], runtime["warmup_repeats"], device)[0]
        graph_time = _timing(lambda: exact_lk_graph(exact_input, n_step=n_step, dt=dt), runtime["timing_repeats"], runtime["warmup_repeats"], device)[0]
        for variant, model, train_seconds in fitted:
            peak = 0.0
            h_timer = lambda: exact_covariance_ratio(evaluation, n_step=n_step, dt=dt)
            if variant["mode"] == "exact":
                prediction, h_hat, timer = reference, h_reference, lambda: exact_estimator(exact_input)
            elif variant["mode"] == "exact_graph":
                prediction, h_hat, timer = graph_reference, h_reference, lambda: exact_lk_graph(exact_input, n_step=n_step, dt=dt)
            elif variant["mode"] == "exact_graph_solve_ex":
                solve_ex_reference = _exact_graph_solve_ex(exact_input, n_step, dt)
                prediction, h_hat, timer = solve_ex_reference, h_reference, lambda: _exact_graph_solve_ex(exact_input, n_step, dt)
            else:
                assert model is not None
                projection = None
                if model.h_method == "projection":
                    rank = model.projection_rank
                    projection = torch.randn(rank, example["time"] - n_step, device=device, dtype=dtype, generator=torch.Generator(device=device).manual_seed(seed + 9)) / rank**0.5
                estimator = LKEstimator(mode="amortized", amortized_model=model).to(device).eval()
                if runtime["compile"]: estimator = torch.compile(estimator, mode=runtime["compile_mode"])
                with torch.no_grad(): prediction, _, h_hat = model(evaluation, projection=projection)
                timer = lambda: estimator(evaluation, projection=projection)
                h_timer = lambda: model.predict_covariance_ratio(evaluation, projection=projection)
                if runtime["compile"]:
                    h_timer = torch.compile(h_timer, mode=runtime["compile_mode"])
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
            inference, wall = _timing(timer, runtime["timing_repeats"], runtime["warmup_repeats"], device)
            h_inference, h_wall = _timing(h_timer, runtime["timing_repeats"], runtime["warmup_repeats"], device)
            if device.type == "cuda": peak = torch.cuda.max_memory_allocated(device) / 2**20
            metrics = _metrics(prediction, reference)
            h_metrics = _h_metrics(h_hat, h_reference) if h_hat is not None else {"h_mse": float("nan"), "magnitude_rank_corr": float("nan")}
            row = {"example": example["name"], "variables": example["variables"], "time": example["time"], "eval_batch_size": evaluation.shape[0], "variant": variant["name"], "mode": variant["mode"], "encoder_type": variant.get("encoder_type", ""), "decoder_type": variant.get("decoder_type", ""), "h_method": getattr(model, "h_method", "exact") if model else "exact", "low_rank": variant.get("low_rank", ""), "h_rank": variant.get("h_rank", ""), "projection_rank": variant.get("projection_rank", ""), "device": str(device), **result_precision_metadata(runtime["dtype"], prediction), "compiled": runtime["compile"] and variant["mode"] == "amortized", "compile_mode": runtime["compile_mode"], "train_seconds": train_seconds, "inference_seconds": inference, "inference_ms": inference * 1000, "inference_ms_wall": wall * 1000, "h_inference_seconds": h_inference, "h_inference_ms": h_inference * 1000, "h_inference_ms_wall": h_wall * 1000, "peak_memory_mb": peak, "speedup_vs_exact_lk": exact_time / max(inference, 1e-12), "speedup_vs_exact_graph_only": graph_time / max(inference, 1e-12), **metrics, "h_mse": h_metrics["h_mse"], "h_magnitude_rank_corr": h_metrics["magnitude_rank_corr"]}
            rows.append(row); print(f"measured {row['example']} / {row['variant']}: {row['inference_ms']:.3f}ms")
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    markdown_path = output_dir / "results.md"
    lines = ["# LK benchmark results", "", f"Device `{device}`, requested dtype `{runtime['dtype']}`.", "", "| Example | Variant | Inference (ms) | H inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | Peak MB |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['example']} | {r['variant']} | {r['inference_ms']:.3f} | {r['h_inference_ms']:.3f} | {r['speedup_vs_exact_graph_only']:.2f}x | {r['magnitude_rank_corr']:.4f} | {r['h_mse']:.5g} | {r['peak_memory_mb']:.1f} |" for r in rows]
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
