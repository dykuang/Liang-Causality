# CPU LK ablation results

Config: `config.yaml`; PyTorch 2.11.0+cpu; device `cpu`.
The `exact_lk` timer includes the repository's variance calculation; `exact_graph_only` isolates graph estimation. Learned timing reports graph inference and does not produce variance.

| Example | Variant | Train (s) | Inference (ms) | Speedup | Graph MSE | Magnitude rank corr. | Strong sign acc. |
|---|---|---:|---:|---:|---:|---:|---:|
| small_oscillator_p2_t256 (p=2, T=256) | exact_lk | 0.00 | 0.931 | 1.00x | 0 | 1.0000 | 1.0000 |
| small_oscillator_p2_t256 (p=2, T=256) | exact_graph_only | 0.00 | 0.292 | 3.19x | 4.967e-18 | 1.0000 | 1.0000 |
| small_oscillator_p2_t256 (p=2, T=256) | tcn_pair_h1 | 2.51 | 1.709 | 0.54x | 0.00144123 | 0.5755 | 0.8125 |
| small_oscillator_p2_t256 (p=2, T=256) | tcn_pair_h0_exact_H | 2.68 | 1.743 | 0.53x | 0.00144424 | 0.6745 | 0.8125 |
| small_oscillator_p2_t256 (p=2, T=256) | tcn_pair_h2 | 2.38 | 1.623 | 0.57x | 0.00175193 | 0.2489 | 0.9375 |
| small_oscillator_p2_t256 (p=2, T=256) | downsample_bilinear_h1 | 1.71 | 0.827 | 1.13x | 0.00125101 | 0.5946 | 0.9375 |
| small_oscillator_p2_t256 (p=2, T=256) | spectral_lowrank_h1 | 1.75 | 0.769 | 1.21x | 0.00127972 | 0.3402 | 0.8750 |
| small_oscillator_p2_t256 (p=2, T=256) | attention_bilinear_residual_h1 | 3.09 | 3.821 | 0.24x | 0.00188326 | 0.4326 | 0.6875 |
| repository_6d_t512 (p=6, T=512) | exact_lk | 0.00 | 1.213 | 1.00x | 0 | 1.0000 | 1.0000 |
| repository_6d_t512 (p=6, T=512) | exact_graph_only | 0.00 | 0.378 | 3.21x | 1.69478e-16 | 1.0000 | 1.0000 |
| repository_6d_t512 (p=6, T=512) | tcn_pair_h1 | 4.74 | 4.020 | 0.30x | 0.102291 | 0.6258 | 0.6944 |
| repository_6d_t512 (p=6, T=512) | tcn_pair_h0_exact_H | 4.61 | 5.060 | 0.24x | 0.0962113 | 0.7451 | 0.6597 |
| repository_6d_t512 (p=6, T=512) | tcn_pair_h2 | 4.74 | 4.928 | 0.25x | 0.112704 | 0.4284 | 0.6528 |
| repository_6d_t512 (p=6, T=512) | downsample_bilinear_h1 | 3.05 | 1.454 | 0.83x | 0.113624 | 0.5914 | 0.6875 |
| repository_6d_t512 (p=6, T=512) | spectral_lowrank_h1 | 3.09 | 1.527 | 0.79x | 0.124962 | 0.5418 | 0.6875 |
| repository_6d_t512 (p=6, T=512) | attention_bilinear_residual_h1 | 6.60 | 12.400 | 0.10x | 0.113767 | 0.5642 | 0.6528 |
| sparse_var_p32_t512 (p=32, T=512) | exact_lk | 0.00 | 1.586 | 1.00x | 0 | 1.0000 | 1.0000 |
| sparse_var_p32_t512 (p=32, T=512) | exact_graph_only | 0.00 | 0.705 | 2.25x | 4.40777e-17 | 1.0000 | 1.0000 |
| sparse_var_p32_t512 (p=32, T=512) | tcn_pair_h1 | 3.91 | 12.019 | 0.13x | 0.0323748 | 0.2266 | 0.5269 |
| sparse_var_p32_t512 (p=32, T=512) | tcn_pair_h0_exact_H | 4.30 | 12.926 | 0.12x | 0.0323741 | 0.7006 | 0.5244 |
| sparse_var_p32_t512 (p=32, T=512) | tcn_pair_h2 | 4.23 | 12.474 | 0.13x | 0.0343556 | -0.0757 | 0.5137 |
| sparse_var_p32_t512 (p=32, T=512) | downsample_bilinear_h1 | 1.52 | 1.703 | 0.93x | 0.0323034 | 0.2202 | 0.5269 |
| sparse_var_p32_t512 (p=32, T=512) | spectral_lowrank_h1 | 1.63 | 1.691 | 0.94x | 0.0319889 | 0.2241 | 0.5269 |
| sparse_var_p32_t512 (p=32, T=512) | attention_bilinear_residual_h1 | 6.36 | 26.906 | 0.06x | 0.0346577 | 0.1384 | 0.4878 |
