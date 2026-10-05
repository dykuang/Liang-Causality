import unittest

import torch

from amortized_lk import AmortizedLK
from train_amortized_lk import train_amortized_step


class AmortizedTrainingStepTests(unittest.TestCase):
    def test_step_applies_probe_and_optional_sampled_h_updates(self):
        torch.manual_seed(13)
        windows = torch.randn(2, 3, 20, dtype=torch.float64)

        for h_method in ("learned_pair", "learned_low_rank", "projection"):
            with self.subTest(h_method=h_method):
                model = AmortizedLK(
                    width=4, h_method=h_method, h_rank=2, projection_rank=4,
                    encoder_type="downsample_mlp", decoder_type="low_rank",
                    low_rank=2,
                ).double().train()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                before = [parameter.detach().clone() for parameter in model.parameters()]

                equation_loss, h_loss = train_amortized_step(
                    model, optimizer, windows, probes=2, ridge_scale=1e-3,
                    loss_mode="plain", probe_distribution="rademacher",
                    generator=torch.Generator().manual_seed(23), h_weight=1.0,
                    h_pairs=4,
                )

                self.assertTrue(torch.isfinite(equation_loss))
                self.assertTrue(torch.isfinite(h_loss))
                self.assertGreater(
                    sum(float((old - new).abs().sum()) for old, new in zip(before, model.parameters())),
                    0.0,
                )
                if h_method == "projection":
                    self.assertEqual(float(h_loss), 0.0)
                else:
                    self.assertGreater(float(h_loss), 0.0)


if __name__ == "__main__":
    unittest.main()
