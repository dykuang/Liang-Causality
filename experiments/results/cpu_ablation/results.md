# LK benchmark results

Device `cpu`, requested dtype `float32`.

| Example | Variant | Inference (ms) | H inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | Peak MB |
|---|---|---:|---:|---:|---:|---:|---:|
| small_oscillator_p2_t256 | exact_lk | 0.268 | 0.034 | 0.24x | 1.0000 | 0 | 0.0 |
| small_oscillator_p2_t256 | exact_graph_only | 0.063 | 0.035 | 1.00x | 1.0000 | 0 | 0.0 |
| small_oscillator_p2_t256 | tcn_pair_h1 | 0.634 | 0.038 | 0.10x | 0.7474 | 0.0080819 | 0.0 |
| small_oscillator_p2_t256 | tcn_pair_h0_exact_H | 0.626 | 0.036 | 0.10x | 0.7896 | 0 | 0.0 |
| small_oscillator_p2_t256 | tcn_pair_h2 | 0.582 | 0.537 | 0.11x | 0.2364 | 0.37408 | 0.0 |
| small_oscillator_p2_t256 | downsample_bilinear_h1 | 0.218 | 0.039 | 0.29x | 0.6023 | 0.0080819 | 0.0 |
| small_oscillator_p2_t256 | spectral_lowrank_h1 | 0.248 | 0.040 | 0.26x | 0.5088 | 0.0080819 | 0.0 |
| small_oscillator_p2_t256 | attention_bilinear_residual_h1 | 1.036 | 0.039 | 0.06x | 0.3479 | 0.0080819 | 0.0 |
| repository_6d_t512 | exact_lk | 0.349 | 0.051 | 0.26x | 1.0000 | 0 | 0.0 |
| repository_6d_t512 | exact_graph_only | 0.089 | 0.051 | 1.00x | 1.0000 | 0 | 0.0 |
| repository_6d_t512 | tcn_pair_h1 | 2.091 | 0.070 | 0.04x | 0.5766 | 0.016179 | 0.0 |
| repository_6d_t512 | tcn_pair_h0_exact_H | 2.082 | 0.052 | 0.04x | 0.6892 | 0 | 0.0 |
| repository_6d_t512 | tcn_pair_h2 | 2.015 | 1.944 | 0.04x | 0.4503 | 0.1319 | 0.0 |
| repository_6d_t512 | downsample_bilinear_h1 | 0.307 | 0.070 | 0.29x | 0.5816 | 0.016179 | 0.0 |
| repository_6d_t512 | spectral_lowrank_h1 | 0.361 | 0.069 | 0.25x | 0.5644 | 0.016179 | 0.0 |
| repository_6d_t512 | attention_bilinear_residual_h1 | 2.346 | 0.070 | 0.04x | 0.5813 | 0.016179 | 0.0 |
| sparse_var_p32_t512 | exact_lk | 0.643 | 0.112 | 0.33x | 1.0000 | 0 | 0.0 |
| sparse_var_p32_t512 | exact_graph_only | 0.214 | 0.112 | 0.99x | 1.0000 | 0 | 0.0 |
| sparse_var_p32_t512 | tcn_pair_h1 | 13.343 | 0.149 | 0.02x | 0.2272 | 0.016926 | 0.0 |
| sparse_var_p32_t512 | tcn_pair_h0_exact_H | 13.256 | 0.112 | 0.02x | 0.7004 | 0 | 0.0 |
| sparse_var_p32_t512 | tcn_pair_h2 | 12.969 | 12.631 | 0.02x | -0.0746 | 0.036706 | 0.0 |
| sparse_var_p32_t512 | downsample_bilinear_h1 | 0.558 | 0.148 | 0.38x | 0.2230 | 0.016926 | 0.0 |
| sparse_var_p32_t512 | spectral_lowrank_h1 | 0.607 | 0.148 | 0.35x | 0.2263 | 0.016926 | 0.0 |
| sparse_var_p32_t512 | attention_bilinear_residual_h1 | 6.586 | 0.149 | 0.03x | 0.2229 | 0.016926 | 0.0 |
