# LK benchmark results

Device `cpu`, requested dtype `float32`.

| Example | Variant | Inference (ms) | H inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | ρ symmetry err. | ρ diagonal err. | Peak MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small_oscillator_p2_t256 | exact_lk | 0.277 | 0.036 | 0.23x | 1.0000 | 0 | NA | NA | 0.0 |
| small_oscillator_p2_t256 | exact_graph_only | 0.065 | 0.036 | 0.99x | 1.0000 | 0 | NA | NA | 0.0 |
| small_oscillator_p2_t256 | tcn_pair_h1 | 0.631 | 0.039 | 0.10x | 0.7474 | 0.0080819 | NA | NA | 0.0 |
| small_oscillator_p2_t256 | tcn_pair_h0_exact_H | 0.624 | 0.035 | 0.10x | 0.7896 | 0 | NA | NA | 0.0 |
| small_oscillator_p2_t256 | tcn_pair_h2 | 0.577 | 0.524 | 0.11x | 0.2364 | 0.37408 | NA | NA | 0.0 |
| small_oscillator_p2_t256 | downsample_bilinear_h1 | 0.220 | 0.039 | 0.29x | 0.6023 | 0.0080819 | NA | NA | 0.0 |
| small_oscillator_p2_t256 | spectral_lowrank_h1 | 0.253 | 0.040 | 0.26x | 0.5088 | 0.0080819 | NA | NA | 0.0 |
| small_oscillator_p2_t256 | attention_bilinear_residual_h1 | 1.009 | 0.040 | 0.06x | 0.3479 | 0.0080819 | NA | NA | 0.0 |
| repository_6d_t512 | exact_lk | 0.350 | 0.052 | 0.26x | 1.0000 | 0 | NA | NA | 0.0 |
| repository_6d_t512 | exact_graph_only | 0.091 | 0.052 | 1.01x | 1.0000 | 0 | NA | NA | 0.0 |
| repository_6d_t512 | tcn_pair_h1 | 2.116 | 0.072 | 0.04x | 0.5766 | 0.016179 | NA | NA | 0.0 |
| repository_6d_t512 | tcn_pair_h0_exact_H | 2.087 | 0.052 | 0.04x | 0.6892 | 0 | NA | NA | 0.0 |
| repository_6d_t512 | tcn_pair_h2 | 2.041 | 1.984 | 0.04x | 0.4503 | 0.1319 | NA | NA | 0.0 |
| repository_6d_t512 | downsample_bilinear_h1 | 0.317 | 0.071 | 0.29x | 0.5816 | 0.016179 | NA | NA | 0.0 |
| repository_6d_t512 | spectral_lowrank_h1 | 0.367 | 0.071 | 0.25x | 0.5644 | 0.016179 | NA | NA | 0.0 |
| repository_6d_t512 | attention_bilinear_residual_h1 | 2.368 | 0.070 | 0.04x | 0.5813 | 0.016179 | NA | NA | 0.0 |
| sparse_var_p32_t512 | exact_lk | 0.645 | 0.119 | 0.33x | 1.0000 | 0 | NA | NA | 0.0 |
| sparse_var_p32_t512 | exact_graph_only | 0.218 | 0.117 | 0.98x | 1.0000 | 0 | NA | NA | 0.0 |
| sparse_var_p32_t512 | tcn_pair_h1 | 5.622 | 0.152 | 0.04x | 0.2272 | 0.016926 | NA | NA | 0.0 |
| sparse_var_p32_t512 | tcn_pair_h0_exact_H | 5.563 | 0.115 | 0.04x | 0.7004 | 0 | NA | NA | 0.0 |
| sparse_var_p32_t512 | tcn_pair_h2 | 5.713 | 5.306 | 0.04x | -0.0746 | 0.036706 | NA | NA | 0.0 |
| sparse_var_p32_t512 | downsample_bilinear_h1 | 0.554 | 0.151 | 0.38x | 0.2230 | 0.016926 | NA | NA | 0.0 |
| sparse_var_p32_t512 | spectral_lowrank_h1 | 0.590 | 0.152 | 0.36x | 0.2263 | 0.016926 | NA | NA | 0.0 |
| sparse_var_p32_t512 | attention_bilinear_residual_h1 | 6.429 | 0.156 | 0.03x | 0.2229 | 0.016926 | NA | NA | 0.0 |
