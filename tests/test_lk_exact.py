import unittest
from unittest.mock import patch

import torch

from Causality_est_torch import LiangCausalityEstimator, causal_est_matrix
from evaluate_amortized_lk import exact_lk_graph
from lk_exact import ExactGraphResult, estimate_exact_graph
from lk_statistics import prepare_lk_statistics


class ExactGraphTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(23)
        self.x = torch.randn(3, 4, 40, dtype=torch.float64, generator=generator)

    def test_graph_only_result_matches_batched_exact_estimator(self):
        n_step, dt = 2, 0.5
        result = estimate_exact_graph(
            prepare_lk_statistics(self.x, n_step=n_step, dt=dt)
        )
        expected_graph, _, _ = LiangCausalityEstimator(n_step=n_step, dt=dt)(self.x)

        self.assertIsInstance(result, ExactGraphResult)
        self.assertEqual(result.effective_dtype, torch.float64)
        self.assertEqual(tuple(result.graph.shape), (3, 4, 4))
        self.assertEqual(tuple(result.coefficients.shape), (3, 4, 4))
        self.assertEqual(tuple(result.covariance_ratio.shape), (3, 4, 4))
        torch.testing.assert_close(result.graph, expected_graph, atol=1e-10, rtol=1e-10)

    def test_batched_estimator_promotes_low_precision_inputs(self):
        n_step, dt = 2, 0.5
        estimator = LiangCausalityEstimator(n_step=n_step, dt=dt)

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                expected = estimator(self.x.to(dtype=dtype).float())
                actual = estimator(self.x.to(dtype=dtype))

                for result, expected_result in zip(actual, expected):
                    self.assertEqual(result.dtype, torch.float32)
                    torch.testing.assert_close(
                        result, expected_result, atol=1e-5, rtol=1e-5
                    )

    def test_batched_estimator_outputs_match_single_window_estimator(self):
        n_step, dt = 2, 0.5
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        for device in devices:
            with self.subTest(device=device):
                x = self.x.to(device)
                graph, variance, normalized_graph = LiangCausalityEstimator(
                    n_step=n_step, dt=dt
                )(x)

                self.assertEqual(tuple(graph.shape), (3, 4, 4))
                self.assertEqual(tuple(variance.shape), (3, 4, 4))
                self.assertEqual(tuple(normalized_graph.shape), (3, 4, 4))
                for batch_index in range(x.shape[0]):
                    expected = causal_est_matrix(
                        x[batch_index], n_step=n_step, dt=dt
                    )
                    for actual_result, expected_result in zip(
                        (graph, variance, normalized_graph), expected
                    ):
                        torch.testing.assert_close(
                            actual_result[batch_index], expected_result
                        )

    def test_pseudoinverse_fallback_is_limited_to_singular_windows(self):
        x = self.x[:2].clone()
        x[1, 3] = 2 * x[1, 0]
        statistics = prepare_lk_statistics(x)

        with patch("torch.linalg.pinv", wraps=torch.linalg.pinv) as pinv:
            result = estimate_exact_graph(statistics)

        self.assertEqual(pinv.call_count, 1)
        self.assertEqual(tuple(pinv.call_args.args[0].shape), (1, 4, 4))
        self.assertTrue(torch.isfinite(result.graph).all())
        self.assertTrue(torch.isfinite(result.coefficients).all())
        expected_regular = torch.linalg.solve(
            statistics.covariance[0], statistics.cross_covariance[0]
        )
        expected_singular = torch.linalg.pinv(
            statistics.covariance[1]
        ) @ statistics.cross_covariance[1]
        torch.testing.assert_close(result.coefficients[0], expected_regular)
        torch.testing.assert_close(result.coefficients[1], expected_singular)

    def test_direct_solve_fallback_is_limited_to_failed_windows(self):
        x = self.x[:2].clone()
        x[1, 3] = 2 * x[1, 0]
        statistics = prepare_lk_statistics(x)
        expected = torch.linalg.pinv(
            statistics.covariance[1]
        ) @ statistics.cross_covariance[1]

        with patch("torch.linalg.solve", return_value=expected.unsqueeze(0)) as solve:
            result = estimate_exact_graph(
                statistics, singular_fallback="solve"
            )

        self.assertEqual(solve.call_count, 1)
        self.assertEqual(tuple(solve.call_args.args[0].shape), (1, 4, 4))
        torch.testing.assert_close(result.coefficients[1], expected)

    def test_evaluation_helper_remains_a_tuple_adapter(self):
        result = estimate_exact_graph(prepare_lk_statistics(self.x))
        graph, coefficients, ratio = exact_lk_graph(self.x)

        torch.testing.assert_close(graph, result.graph)
        torch.testing.assert_close(coefficients, result.coefficients)
        torch.testing.assert_close(ratio, result.covariance_ratio)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_low_precision_statistics_return_float32_results(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = self.x.cuda().to(dtype=dtype)
                statistics = prepare_lk_statistics(x)
                full_precision_statistics = prepare_lk_statistics(x.float())
                result = estimate_exact_graph(statistics)
                self.assertEqual(result.effective_dtype, torch.float32)
                self.assertEqual(statistics.state.dtype, torch.float32)
                torch.testing.assert_close(statistics.state, full_precision_statistics.state)
                torch.testing.assert_close(
                    statistics.derivative, full_precision_statistics.derivative
                )
                self.assertEqual(result.graph.dtype, torch.float32)
                self.assertEqual(result.coefficients.dtype, torch.float32)
                self.assertEqual(result.covariance_ratio.dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_compatibility_helper_promotes_cuda_low_precision_before_preparation(self):
        x = self.x.cuda().to(dtype=torch.float16)
        result = exact_lk_graph(x)
        expected = estimate_exact_graph(prepare_lk_statistics(x.float()))

        for actual, expected_tensor in zip(result, (
            expected.graph,
            expected.coefficients,
            expected.covariance_ratio,
        )):
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected_tensor)


if __name__ == "__main__":
    unittest.main()
