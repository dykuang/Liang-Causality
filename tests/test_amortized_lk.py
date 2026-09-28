import unittest
from unittest.mock import patch

import torch

from amortized_lk import (
    AmortizedLK,
    exact_covariance_ratio,
    random_projection_covariance_ratio,
    sampled_covariance_ratio_loss,
    stochastic_probe_residual_loss,
)
from lk_estimators import LKEstimator


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
