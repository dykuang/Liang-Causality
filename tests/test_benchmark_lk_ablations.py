import csv
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from benchmark_lk_ablations import (
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
    def test_precision_metadata_uses_computed_dtype_for_each_variant_kind(self):
        exact = result_precision_metadata(
            "bfloat16", torch.zeros((), dtype=torch.float32)
        )
        amortized = result_precision_metadata(
            "bfloat16", torch.zeros((), dtype=torch.bfloat16)
        )

        self.assertEqual(exact, {"requested_dtype": "bfloat16", "effective_dtype": "float32"})
        self.assertEqual(amortized, {"requested_dtype": "bfloat16", "effective_dtype": "bfloat16"})

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
                ],
            }
            source_config.write_text(yaml.safe_dump(config), encoding="utf-8")

            csv_path, _ = run(source_config)

            with csv_path.open(newline="", encoding="utf-8") as handle:
                first_rows = list(csv.DictReader(handle))
            copied_config = output_dir / "config.yaml"
            self.assertEqual(copied_config.read_bytes(), source_config.read_bytes())
            self.assertEqual(
                [(row["variant"], row["requested_dtype"], row["effective_dtype"]) for row in first_rows],
                [
                    ("exact_graph_only", "float32", "float32"),
                    ("amortized_exact_h", "float32", "float32"),
                ],
            )

            reproduced_csv_path, _ = run(copied_config)
            with reproduced_csv_path.open(newline="", encoding="utf-8") as handle:
                reproduced_rows = list(csv.DictReader(handle))

            timing_fields = {
                "train_seconds",
                "inference_seconds",
                "inference_ms",
                "inference_ms_wall",
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


if __name__ == "__main__":
    unittest.main()
