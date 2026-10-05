import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import torch

from lk_statistics import LKStatistics, LKSufficientStatistics, prepare_lk_statistics


class LKStatisticsTests(unittest.TestCase):
    def test_legacy_statistics_name_is_public_alias(self):
        self.assertIs(LKStatistics, LKSufficientStatistics)
        prepared = prepare_lk_statistics(torch.randn(1, 2, 8))
        self.assertIsInstance(prepared, LKSufficientStatistics)

    def test_preparation_centers_state_and_finite_difference(self):
        x = torch.tensor(
            [[[0, 2, 4, 6, 10], [10, 8, 4, 2, 0]]], dtype=torch.float64
        )

        statistics = prepare_lk_statistics(x, n_step=2, dt=0.5)

        expected_state = torch.tensor(
            [[[-2, 0, 2], [8 / 3, 2 / 3, -10 / 3]]], dtype=torch.float64
        )
        expected_derivative = torch.tensor(
            [[[-2 / 3, -2 / 3, 4 / 3], [-2 / 3, -2 / 3, 4 / 3]]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(statistics.state, expected_state)
        torch.testing.assert_close(statistics.derivative, expected_derivative)
        with self.assertRaises(FrozenInstanceError):
            statistics.n_step = 1

    def test_dense_statistics_are_lazy_and_cached(self):
        x = torch.tensor(
            [[[0, 2, 4, 6, 10], [10, 8, 4, 2, 0]]], dtype=torch.float64
        )

        with patch("torch.matmul", wraps=torch.matmul) as matmul:
            statistics = prepare_lk_statistics(x, n_step=2, dt=0.5)
            _ = statistics.state, statistics.derivative
            self.assertEqual(matmul.call_count, 0)

            covariance_ratio = statistics.covariance_ratio
            self.assertEqual(matmul.call_count, 1)
            self.assertIs(statistics.covariance_ratio, covariance_ratio)
            covariance = statistics.covariance
            self.assertIs(statistics.covariance, covariance)
            self.assertEqual(matmul.call_count, 1)

            cross_covariance = statistics.cross_covariance
            self.assertIs(statistics.cross_covariance, cross_covariance)
            self.assertEqual(matmul.call_count, 2)

        torch.testing.assert_close(
            covariance,
            torch.tensor([[[4, -6], [-6, 28 / 3]]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            cross_covariance,
            torch.tensor([[[2, 2], [-10 / 3, -10 / 3]]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            covariance_ratio,
            torch.tensor([[[1, -9 / 14], [-1.5, 1]]], dtype=torch.float64),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cpu_and_cuda_statistics_agree(self):
        for dtype, tolerance in (
            (torch.float32, 1e-5),
            (torch.float64, 1e-11),
        ):
            generator = torch.Generator().manual_seed(19)
            x = torch.randn(2, 3, 24, dtype=dtype, generator=generator)
            cpu = prepare_lk_statistics(x, n_step=3, dt=0.25)
            cuda = prepare_lk_statistics(x.cuda(), n_step=3, dt=0.25)

            for property_name in (
                "state",
                "derivative",
                "covariance",
                "cross_covariance",
                "covariance_ratio",
            ):
                torch.testing.assert_close(
                    getattr(cuda, property_name).cpu(),
                    getattr(cpu, property_name),
                    atol=tolerance,
                    rtol=tolerance,
                )


if __name__ == "__main__":
    unittest.main()
