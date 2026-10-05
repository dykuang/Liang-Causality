import csv
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from lk_exact import estimate_exact_graph
from lk_statistics import prepare_lk_statistics

from benchmark_lk_ablations import (
    _example_batch,
    _exact_graph_solve_ex,
    _h_metrics,
    result_precision_metadata,
    run,
    validate_config,
)


class BenchmarkConfigTests(unittest.TestCase):
    def test_cpu_config_rejects_mixed_precision(self):
        config = {
            "runtime": {"device": "cpu", "dtype": "bfloat16"},
            "training": {},
            "variants": [],
        }

        with self.assertRaisesRegex(ValueError, "mixed precision is supported only on CUDA"):
            validate_config(config)

    def test_example_batch_honors_requested_dtype_for_all_families(self):
        for generator_name, variables in (("oscillator", 2), ("repository6d", 6)):
            with self.subTest(generator=generator_name):
                batch = _example_batch(
                    {"generator": generator_name, "variables": variables, "time": 12},
                    2,
                    device=torch.device("cpu"),
                    generator=torch.Generator().manual_seed(7),
                    dtype=torch.float64,
                )

                self.assertEqual(batch.dtype, torch.float64)

    def test_gpu_h_variant_config_covers_planned_rank_sweep(self):
        config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "experiments/gpu_h_variants.yaml").read_text(encoding="utf-8")
        )
        variants = config["variants"]
        projection_ranks = sorted(
            variant["projection_rank"]
            for variant in variants
            if variant.get("h_method") == "projection"
        )
        low_rank_h_ranks = sorted(
            variant["h_rank"]
            for variant in variants
            if variant.get("h_method") == "learned_low_rank"
        )

        self.assertEqual(projection_ranks, [8, 16, 32, 64])
        self.assertEqual(low_rank_h_ranks, [4, 8, 16, 32])

    def test_validation_canonicalizes_legacy_learned_h_method(self):
        config = {
            "runtime": {"device": "cpu", "dtype": "float32"},
            "training": {},
            "variants": [{"mode": "amortized", "h_method": "learned"}],
        }

        validated = validate_config(config)

        self.assertEqual(validated["variants"][0]["h_method"], "learned_pair")

    def test_validation_resolves_requested_dtype_without_losing_runtime_config(self):
        config = {
            "runtime": {"device": "cpu", "dtype": "float32"},
            "training": {},
            "variants": [],
        }

        validated = validate_config(config)

        self.assertEqual(validated["runtime"]["dtype"], "float32")
        self.assertIs(validated["dtype"], torch.float32)


class BenchmarkResultTests(unittest.TestCase):
    def test_h_magnitude_rank_correlation_ignores_diagonal(self):
        target = torch.tensor([[[1.0, 0.2, 0.4], [0.3, 1.0, 0.5], [0.6, 0.7, 1.0]]])
        prediction = target.clone()
        baseline = _h_metrics(prediction, target)["magnitude_rank_corr"]
        prediction.diagonal(dim1=-2, dim2=-1).copy_(torch.tensor([[100.0, -80.0, 60.0]]))

        changed_diagonal = _h_metrics(prediction, target)["magnitude_rank_corr"]

        self.assertEqual(baseline, 1.0)
        self.assertEqual(changed_diagonal, baseline)

    def test_precision_metadata_uses_computed_dtype_for_each_variant_kind(self):
        exact = result_precision_metadata(
            "bfloat16", torch.zeros((), dtype=torch.float32)
        )
        amortized = result_precision_metadata(
            "bfloat16", torch.zeros((), dtype=torch.bfloat16)
        )

        self.assertEqual(exact, {"requested_dtype": "bfloat16", "effective_dtype": "float32"})
        self.assertEqual(amortized, {"requested_dtype": "bfloat16", "effective_dtype": "bfloat16"})

    def test_run_canonicalizes_legacy_h_method_in_saved_config_and_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_config = root / "legacy.yaml"
            output_dir = root / "results"
            config = {
                "runtime": {
                    "device": "cpu", "dtype": "float32", "compile": False,
                    "warmup_repeats": 0, "timing_repeats": 1,
                },
                "seed": 19,
                "output_dir": str(output_dir),
                "training": {
                    "steps": 0, "batch_size": 1, "learning_rate": 0.002,
                    "probes": 1, "ridge_scale": 0.001, "loss_mode": "plain",
                    "probe_distribution": "rademacher", "h_pairs": 2,
                    "h_weight": 1.0, "width": 4, "low_rank": 2,
                },
                "examples": [{
                    "name": "tiny", "generator": "stable_var",
                    "variables": 2, "time": 8, "eval_examples": 1,
                    "training_steps": 0, "training_batch_size": 1,
                }],
                "variants": [{
                    "name": "legacy_learned", "mode": "amortized",
                    "encoder_type": "downsample_mlp", "decoder_type": "low_rank",
                    "low_rank": 2, "h_method": "learned",
                }],
            }
            source_config.write_text(yaml.safe_dump(config), encoding="utf-8")
            original_source = source_config.read_bytes()

            csv_path, _ = run(source_config)

            saved = yaml.safe_load((output_dir / "config.yaml").read_text(encoding="utf-8"))
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(source_config.read_bytes(), original_source)
            self.assertEqual(saved["variants"][0]["h_method"], "learned_pair")
            self.assertEqual(rows[0]["h_method"], "learned_pair")

    def test_rows_and_copied_config_can_be_reproduced_without_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_config = root / "tiny.yaml"
            output_dir = root / "results"
            config = {
                "runtime": {
                    "device": "cpu",
                    "dtype": "float32",
                    "compile": False,
                    "warmup_repeats": 0,
                    "timing_repeats": 1,
                },
                "seed": 17,
                "output_dir": str(output_dir),
                "training": {
                    "steps": 0,
                    "batch_size": 1,
                    "learning_rate": 0.002,
                    "probes": 1,
                    "ridge_scale": 0.001,
                    "loss_mode": "plain",
                    "probe_distribution": "rademacher",
                    "h_pairs": 2,
                    "h_weight": 1.0,
                    "width": 8,
                    "low_rank": 2,
                },
                "examples": [
                    {
                        "name": "tiny",
                        "generator": "stable_var",
                        "variables": 2,
                        "time": 8,
                        "eval_examples": 1,
                        "training_steps": 0,
                        "training_batch_size": 1,
                    }
                ],
                "variants": [
                    {"name": "exact_graph_only", "mode": "exact_graph"},
                    {
                        "name": "amortized_exact_h",
                        "mode": "amortized",
                        "encoder_type": "downsample_mlp",
                        "decoder_type": "low_rank",
                        "low_rank": 2,
                        "h_method": "exact",
                    },
                    {
                        "name": "amortized_exact_h_same_arch",
                        "mode": "amortized",
                        "encoder_type": "downsample_mlp",
                        "decoder_type": "low_rank",
                        "low_rank": 2,
                        "h_method": "exact",
                    },
                ],
            }
            source_config.write_text(yaml.safe_dump(config), encoding="utf-8")

            csv_path, _ = run(source_config)

            with csv_path.open(newline="", encoding="utf-8") as handle:
                first_rows = list(csv.DictReader(handle))
            copied_config = output_dir / "config.yaml"
            self.assertEqual(copied_config.read_bytes(), source_config.read_bytes())
            self.assertTrue(all(float(row["h_inference_ms"]) >= 0 for row in first_rows))
            self.assertTrue(all(float(row["h_inference_ms_wall"]) >= 0 for row in first_rows))
            markdown = (output_dir / "results.md").read_text(encoding="utf-8")
            self.assertIn("H inference (ms)", markdown)
            markdown_lines = markdown.splitlines()
            self.assertEqual(markdown_lines[4].count("|"), markdown_lines[5].count("|"))
            self.assertEqual(
                [(row["variant"], row["requested_dtype"], row["effective_dtype"]) for row in first_rows],
                [
                    ("exact_graph_only", "float32", "float32"),
                    ("amortized_exact_h", "float32", "float32"),
                    ("amortized_exact_h_same_arch", "float32", "float32"),
                ],
            )
            shared_seed_metrics = (
                "graph_mse", "graph_mae", "magnitude_rank_corr",
                "strong_sign_accuracy", "support_f1_q75", "h_mse",
                "h_magnitude_rank_corr",
            )
            for metric in shared_seed_metrics:
                self.assertEqual(first_rows[1][metric], first_rows[2][metric], metric)

            reproduced_csv_path, _ = run(copied_config)
            with reproduced_csv_path.open(newline="", encoding="utf-8") as handle:
                reproduced_rows = list(csv.DictReader(handle))

            timing_fields = {
                "train_seconds",
                "inference_seconds",
                "inference_ms",
                "inference_ms_wall",
                "h_inference_seconds",
                "h_inference_ms",
                "h_inference_ms_wall",
                "speedup_vs_exact_lk",
                "speedup_vs_exact_graph_only",
            }
            self.assertEqual(
                [
                    {key: value for key, value in row.items() if key not in timing_fields}
                    for row in reproduced_rows
                ],
                [
                    {key: value for key, value in row.items() if key not in timing_fields}
                    for row in first_rows
                ],
            )

    def test_solve_ex_benchmark_uses_shared_exact_graph_kernel(self):
        x = torch.randn(2, 4, 32, dtype=torch.float64)
        statistics = prepare_lk_statistics(x, n_step=2, dt=0.5)

        expected = estimate_exact_graph(
            statistics, singular_fallback="solve"
        ).graph
        actual = _exact_graph_solve_ex(x, n_step=2, dt=0.5)

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
