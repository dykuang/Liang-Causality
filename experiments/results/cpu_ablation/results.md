# LK benchmark results

Device `cpu`, requested dtype `float32`.

| Example | Variant | Inference (ms) | H inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | Peak MB |
|---|---|---:|---:|---:|---:|---:|---:|
| small_oscillator_p2_t256 | exact_lk | 0.273 | 0.035 | 0.23x | 1.0000 | 0 | 0.0 |
| small_oscillator_p2_t256 | exact_graph_only | 0.064 | 0.035 | 0.99x | 1.0000 | 0 | 0.0 |
| small_oscillator_p2_t256 | tcn_pair_h1 | 0.642 | 0.039 | 0.10x | 0.6085 | 0.0080819 | 0.0 |
| small_oscillator_p2_t256 | tcn_pair_h0_exact_H | 0.638 | 0.036 | 0.10x | 0.6298 | 0 | 0.0 |
| small_oscillator_p2_t256 | tcn_pair_h2 | 0.600 | 0.547 | 0.11x | 0.2060 | 0.37427 | 0.0 |
| small_oscillator_p2_t256 | downsample_bilinear_h1 | 0.217 | 0.038 | 0.29x | 0.5707 | 0.0080819 | 0.0 |
| small_oscillator_p2_t256 | spectral_lowrank_h1 | 0.254 | 0.038 | 0.25x | 0.2819 | 0.0080819 | 0.0 |
| small_oscillator_p2_t256 | attention_bilinear_residual_h1 | 1.043 | 0.039 | 0.06x | 0.4370 | 0.0080819 | 0.0 |
| repository_6d_t512 | exact_lk | 0.341 | 0.052 | 0.27x | 1.0000 | 0 | 0.0 |
| repository_6d_t512 | exact_graph_only | 0.090 | 0.051 | 1.01x | 1.0000 | 0 | 0.0 |
| repository_6d_t512 | tcn_pair_h1 | 2.104 | 0.069 | 0.04x | 0.6145 | 0.016179 | 0.0 |
| repository_6d_t512 | tcn_pair_h0_exact_H | 2.079 | 0.052 | 0.04x | 0.7231 | 0 | 0.0 |
| repository_6d_t512 | tcn_pair_h2 | 1.997 | 1.956 | 0.05x | 0.4928 | 0.086462 | 0.0 |
| repository_6d_t512 | downsample_bilinear_h1 | 0.298 | 0.069 | 0.30x | 0.5787 | 0.016179 | 0.0 |
| repository_6d_t512 | spectral_lowrank_h1 | 0.357 | 0.068 | 0.25x | 0.5692 | 0.016179 | 0.0 |
| repository_6d_t512 | attention_bilinear_residual_h1 | 2.392 | 0.070 | 0.04x | 0.5643 | 0.016179 | 0.0 |
| sparse_var_p32_t512 | exact_lk | 0.620 | 0.108 | 0.34x | 1.0000 | 0 | 0.0 |
| sparse_var_p32_t512 | exact_graph_only | 0.210 | 0.111 | 1.00x | 1.0000 | 0 | 0.0 |
| sparse_var_p32_t512 | tcn_pair_h1 | 5.710 | 0.147 | 0.04x | 0.2267 | 0.016926 | 0.0 |
| sparse_var_p32_t512 | tcn_pair_h0_exact_H | 5.598 | 0.112 | 0.04x | 0.7004 | 0 | 0.0 |
| sparse_var_p32_t512 | tcn_pair_h2 | 5.814 | 5.252 | 0.04x | -0.0753 | 0.036749 | 0.0 |
| sparse_var_p32_t512 | downsample_bilinear_h1 | 0.556 | 0.145 | 0.38x | 0.2109 | 0.016926 | 0.0 |
| sparse_var_p32_t512 | spectral_lowrank_h1 | 0.582 | 0.142 | 0.36x | 0.2272 | 0.016926 | 0.0 |
| sparse_var_p32_t512 | attention_bilinear_residual_h1 | 6.540 | 0.148 | 0.03x | 0.1522 | 0.016926 | 0.0 |
