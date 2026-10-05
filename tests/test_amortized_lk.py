import unittest
from unittest.mock import patch

import torch

from amortized_lk import (
    AmortizedLK,
    exact_covariance_ratio,
    random_projection_covariance_ratio,
    sampled_covariance_ratio_loss,
    sampled_covariance_ratio_pair_loss,
    stochastic_probe_residual_loss,
)
from lk_estimators import LKEstimator
from lk_statistics import LKStatistics, prepare_lk_statistics


class _MatrixFreeLKStatistics(LKStatistics):
    @property
    def covariance(self):
        raise AssertionError("matrix-free helper requested covariance")

    @property
    def cross_covariance(self):
        raise AssertionError("matrix-free helper requested cross-covariance")


class _CovarianceOnlyLKStatistics(LKStatistics):
    covariance_reads = 0
    covariance_ratio_reads = 0

    @property
    def covariance(self):
        type(self).covariance_reads += 1
        return super().covariance

    @property
    def covariance_ratio(self):
        type(self).covariance_ratio_reads += 1
        return super().covariance_ratio

    @property
    def cross_covariance(self):
        raise AssertionError("exact H requested cross-covariance")


class AmortizedLKTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(3, 4, 48, dtype=torch.float64)
        self.projection = torch.randn(12, 47, dtype=torch.float64) / 12**0.5

    def test_matrix_free_residual_matches_explicit_equation(self):
        x = self.x
        z = x[..., :-1]
        y = x[..., 1:] - x[..., :-1]
        z = z - z.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)
        m = z.shape[-1]
        c = z @ z.transpose(1, 2) / (m - 1)
        d = z @ y.transpose(1, 2) / (m - 1)
        w = torch.randn(3, 4, 4, dtype=x.dtype)
        ridge = 0.03
        expected = ((c + ridge * torch.eye(4, dtype=x.dtype)) @ w - d).square().sum(dim=(-1, -2)).mean()
        actual = stochastic_probe_residual_loss(
            x, w, q=4, ridge=ridge, probe_distribution="orthogonal"
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-10, rtol=1e-10))

    def test_probe_loss_has_gradient_and_no_exact_solve(self):
        model = AmortizedLK(width=8, projection_rank=12).double()
        with patch("torch.linalg.solve", side_effect=AssertionError("exact solve called")):
            _, w_hat, _ = model(self.x, projection=self.projection)
            loss = stochastic_probe_residual_loss(self.x, w_hat, q=4)
            loss.backward()
        self.assertIsNotNone(model.edge_decoder[0].weight.grad)
        self.assertGreater(float(model.edge_decoder[0].weight.grad.norm()), 0.0)

    def test_prepared_statistics_feed_matrix_free_probe_loss(self):
        n_step, dt = 2, 0.5
        devices = [self.x.device]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            with self.subTest(device=device):
                x = self.x.to(device)
                prepared = prepare_lk_statistics(x, n_step=n_step, dt=dt)
                statistics = _MatrixFreeLKStatistics(
                    prepared.state, prepared.derivative, prepared.n_step, prepared.dt
                )
                w = torch.randn(3, 4, 4, dtype=x.dtype, device=device)
                raw_generator = torch.Generator(device=device).manual_seed(19)
                statistics_generator = torch.Generator(device=device).manual_seed(19)

                expected = stochastic_probe_residual_loss(
                    x, w, q=3, n_step=n_step, dt=dt, generator=raw_generator
                )
                actual = stochastic_probe_residual_loss(
                    statistics, w, q=3, generator=statistics_generator
                )

                self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_covariance_ratio_helpers_accept_prepared_statistics(self):
        n_step, dt = 2, 0.5
        prepared = prepare_lk_statistics(self.x, n_step=n_step, dt=dt)
        matrix_free_statistics = _MatrixFreeLKStatistics(
            prepared.state, prepared.derivative, prepared.n_step, prepared.dt
        )
        projection = torch.randn(8, prepared.state.shape[-1], dtype=self.x.dtype)
        exact_expected = exact_covariance_ratio(self.x, n_step=n_step, dt=dt)
        exact_statistics = _CovarianceOnlyLKStatistics(
            prepared.state, prepared.derivative, prepared.n_step, prepared.dt
        )
        _CovarianceOnlyLKStatistics.covariance_reads = 0
        _CovarianceOnlyLKStatistics.covariance_ratio_reads = 0
        exact_actual = exact_covariance_ratio(exact_statistics)
        self.assertTrue(torch.allclose(exact_actual, exact_expected, atol=1e-12, rtol=1e-12))
        self.assertEqual(_CovarianceOnlyLKStatistics.covariance_reads, 1)
        self.assertEqual(_CovarianceOnlyLKStatistics.covariance_ratio_reads, 1)

        projected_expected = random_projection_covariance_ratio(
            self.x, rank=8, n_step=n_step, dt=dt, projection=projection
        )
        projected_actual = random_projection_covariance_ratio(
            matrix_free_statistics, rank=8, projection=projection
        )
        self.assertTrue(torch.allclose(projected_actual, projected_expected, atol=1e-12, rtol=1e-12))

        pairs = torch.tensor([[0, 1], [2, 3], [1, 1]])
        h_hat = torch.randn(3, 4, 4, dtype=self.x.dtype)
        h_pairs = torch.randn(3, pairs.shape[0], dtype=self.x.dtype)
        sampled_expected = sampled_covariance_ratio_loss(
            self.x, h_hat, pairs, n_step=n_step, dt=dt
        )
        sampled_actual = sampled_covariance_ratio_loss(
            matrix_free_statistics, h_hat, pairs
        )
        self.assertTrue(torch.allclose(sampled_actual, sampled_expected, atol=1e-12, rtol=1e-12))
        pair_expected = sampled_covariance_ratio_pair_loss(
            self.x, h_pairs, pairs, n_step=n_step, dt=dt
        )
        pair_actual = sampled_covariance_ratio_pair_loss(
            matrix_free_statistics, h_pairs, pairs
        )
        self.assertTrue(torch.allclose(pair_actual, pair_expected, atol=1e-12, rtol=1e-12))

    def test_free_matrix_residual_identifies_ols_coefficients(self):
        x = self.x[:1]
        z = x[..., :-1]
        y = x[..., 1:] - x[..., :-1]
        z = z - z.mean(dim=-1, keepdim=True)
        y = y - y.mean(dim=-1, keepdim=True)
        c = z @ z.transpose(1, 2)
        d = z @ y.transpose(1, 2)
        learned = torch.nn.Parameter(torch.zeros_like(c))
        optimizer = torch.optim.Adam([learned], lr=0.08)
        for _ in range(500):
            optimizer.zero_grad()
            loss = stochastic_probe_residual_loss(
                x, learned, q=4, ridge=0.0, probe_distribution="orthogonal"
            )
            loss.backward()
            optimizer.step()
        exact = torch.linalg.solve(c, d)
        self.assertTrue(torch.allclose(learned, exact, atol=2e-3, rtol=2e-3))

    def test_h_only_prediction_matches_forward_without_coefficients(self):
        projection = torch.randn(64, self.x.shape[-1] - 1, dtype=self.x.dtype)
        for method in ("exact", "projection", "learned_pair", "learned_low_rank"):
            with self.subTest(method=method):
                model = AmortizedLK(width=8, h_method=method, h_rank=3).double().eval()
                with torch.no_grad():
                    _, _, expected = model(self.x, projection=projection)
                    with patch.object(
                        model, "_coefficients", side_effect=AssertionError("W computed")
                    ):
                        actual = model.predict_covariance_ratio(
                            self.x, projection=projection
                        )

                torch.testing.assert_close(actual, expected)

    def test_projection_h_matches_exact_when_full_rank_identity(self):
        identity = torch.eye(47, dtype=self.x.dtype)
        exact = exact_covariance_ratio(self.x)
        projected = random_projection_covariance_ratio(
            self.x, rank=47, projection=identity
        )
        self.assertTrue(torch.allclose(projected, exact, atol=1e-12, rtol=1e-12))
        model = AmortizedLK(width=8, h_method="exact").double().eval()
        _, _, h0 = model(self.x, projection=self.projection)
        self.assertTrue(torch.allclose(h0, exact, atol=1e-12, rtol=1e-12))

    def test_probe_and_residual_variants_are_finite(self):
        w = torch.randn(3, 4, 4, dtype=self.x.dtype)
        for distribution in ("rademacher", "gaussian", "orthogonal"):
            for mode in ("plain", "normalized", "jacobi"):
                value = stochastic_probe_residual_loss(
                    self.x,
                    w,
                    q=4,
                    ridge_scale=1e-3,
                    loss_mode=mode,
                    probe_distribution=distribution,
                )
                self.assertTrue(torch.isfinite(value))

    def test_scaling_offset_and_permutation_structure(self):
        model = AmortizedLK(width=8, projection_rank=12).double().eval()
        with torch.no_grad():
            base = model(self.x, projection=self.projection)[0]
            scales = torch.tensor([0.02, -0.4, 3.0, 20.0], dtype=self.x.dtype)
            scaled = model(self.x * scales[None, :, None], projection=self.projection)[0]
            shifted = model(self.x + torch.randn(3, 4, 1, dtype=self.x.dtype), projection=self.projection)[0]
            permutation = torch.tensor([2, 0, 3, 1])
            permuted = model(self.x[:, permutation], projection=self.projection)[0]
        self.assertTrue(torch.allclose(base, scaled, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(base, shifted, atol=1e-10, rtol=1e-10))
        expected_permuted = base[:, permutation][:, :, permutation]
        self.assertTrue(torch.allclose(expected_permuted, permuted, atol=1e-10, rtol=1e-10))

    def test_sampled_h_supervision_is_differentiable(self):
        model = AmortizedLK(width=8, h_method="learned").double()
        _, _, h_hat = model(self.x)
        pairs = torch.tensor([[0, 1], [2, 3], [1, 1]])
        loss = sampled_covariance_ratio_loss(self.x, h_hat, pairs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.h_decoder[0].weight.grad)

    def test_exact_h_is_trainable_and_never_solves(self):
        model = AmortizedLK(width=8, h_method="exact").double().train()
        with patch("torch.linalg.solve", side_effect=AssertionError("exact solve called")), patch("torch.linalg.pinv", side_effect=AssertionError("pinv called")):
            graph, _, h_hat = model(self.x)
        self.assertTrue(torch.allclose(h_hat, exact_covariance_ratio(self.x), atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.isfinite(graph).all())

    def test_low_rank_h_structure_pair_scoring_and_gradients(self):
        model = AmortizedLK(width=8, h_method="learned_low_rank", h_rank=3).double()
        graph, _, h_hat = model(self.x)
        self.assertEqual(tuple(h_hat.shape), (3, 4, 4))
        centered = self.x - self.x.mean(-1, keepdim=True)
        scales = centered.square().mean(-1).sqrt()
        anchor = centered.gather(-1, centered.abs().argmax(dim=-1, keepdim=True)).squeeze(-1)
        signed_scales = scales * torch.where(anchor < 0, -torch.ones_like(anchor), torch.ones_like(anchor))
        recovered_rho = h_hat * signed_scales.unsqueeze(-2) / signed_scales.unsqueeze(-1)
        rho = model.predict_h_correlation(self.x)
        torch.testing.assert_close(rho, recovered_rho, atol=1e-10, rtol=1e-10)
        self.assertTrue(torch.allclose(rho, rho.transpose(-1, -2), atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(torch.diagonal(rho, dim1=-2, dim2=-1), torch.ones(3, 4, dtype=self.x.dtype), atol=1e-10))
        pairs = torch.tensor([[0, 1], [2, 3], [1, 1]])
        selected = model.score_h_pairs(self.x, pairs)
        self.assertTrue(torch.allclose(selected, h_hat[:, pairs[:, 0], pairs[:, 1]], atol=1e-10, rtol=1e-10))
        loss = sampled_covariance_ratio_pair_loss(self.x, selected, pairs)
        loss.backward()
        self.assertGreater(float(model.low_rank_h_head.network[0].weight.grad.norm()), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_low_rank_correlation_structure_is_preserved_in_bfloat16(self):
        model = AmortizedLK(
            width=8, h_method="learned_low_rank", h_rank=3
        ).cuda().to(dtype=torch.bfloat16).eval()
        with torch.no_grad():
            model.low_rank_h_head.network[-1].weight.mul_(1e-2)
            model.low_rank_h_head.network[-1].bias.zero_()
        x = self.x.cuda().to(dtype=torch.bfloat16)
        channel_scales = torch.tensor(
            [0.001, 0.002, 0.004, 0.006], device=x.device, dtype=x.dtype
        )
        scaled_x = x * channel_scales[None, :, None]
        pairs = torch.tensor([[0, 0], [1, 1], [2, 2]], device=x.device)

        with torch.no_grad():
            rho = model.predict_h_correlation(x)
            h = model.predict_covariance_ratio(x)
            scaled_h = model.predict_covariance_ratio(scaled_x)
            selected = model.score_h_pairs(x, pairs)

        rho_diagonal = torch.diagonal(rho, dim1=-2, dim2=-1)
        h_diagonal = torch.diagonal(h, dim1=-2, dim2=-1)
        self.assertLess(float((rho_diagonal - 1).abs().max()), 1e-3)
        self.assertLess(float((h_diagonal - 1).abs().max()), 1e-3)
        expected_scaled_h = (
            h * channel_scales.view(1, -1, 1) / channel_scales.view(1, 1, -1)
        )
        torch.testing.assert_close(
            scaled_h, expected_scaled_h, atol=2e-3, rtol=2e-2
        )
        torch.testing.assert_close(
            selected, h[:, pairs[:, 0], pairs[:, 1]], atol=1e-3, rtol=1e-3
        )

    def test_exact_and_low_rank_h_preserve_scaling_and_permutation(self):
        scales = torch.tensor([0.02, -0.4, 3.0, 20.0], dtype=self.x.dtype)
        permutation = torch.tensor([2, 0, 3, 1])
        for method in ("exact", "learned_low_rank"):
            model = AmortizedLK(width=8, h_method=method, h_rank=3).double().eval()
            with torch.no_grad():
                base = model(self.x)[0]
                scaled = model(self.x * scales[None, :, None])[0]
                permuted = model(self.x[:, permutation])[0]
            self.assertTrue(torch.allclose(base, scaled, atol=1e-10, rtol=1e-10), method)
            self.assertTrue(torch.allclose(base[:, permutation][:, :, permutation], permuted, atol=1e-10, rtol=1e-10), method)

    def test_coefficient_only_forward(self):
        model = AmortizedLK(width=8, h_method="exact").double()
        graph, w_hat, h_hat = model(self.x, compute_graph=False)
        self.assertIsNone(graph)
        self.assertIsNone(h_hat)
        self.assertEqual(tuple(w_hat.shape), (3, 4, 4))

    def test_estimator_selector_and_architecture_variants(self):
        exact = LKEstimator(mode="exact")(self.x.float())
        self.assertEqual(tuple(exact.graph.shape), (3, 4, 4))
        self.assertIsNotNone(exact.variance)

        learned = LKEstimator(
            mode="amortized",
            amortized_options={"width": 8, "projection_rank": 12},
        ).double()(self.x, projection=self.projection)
        self.assertEqual(tuple(learned.graph.shape), (3, 4, 4))
        self.assertIsNone(learned.variance)

        for encoder in ("tcn", "downsample_mlp", "spectral", "attention"):
            for decoder in ("pair_mlp", "bilinear", "low_rank", "bilinear_residual"):
                model = AmortizedLK(
                    width=8,
                    projection_rank=12,
                    encoder_type=encoder,
                    decoder_type=decoder,
                    low_rank=4,
                ).double().eval()
                with torch.no_grad():
                    graph, coefficients, ratio = model(
                        self.x[:1], projection=self.projection
                    )
                self.assertEqual(tuple(graph.shape), (1, 4, 4), (encoder, decoder))
                self.assertEqual(tuple(coefficients.shape), (1, 4, 4))
                self.assertEqual(tuple(ratio.shape), (1, 4, 4))


if __name__ == "__main__":
    unittest.main()
