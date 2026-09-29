# LK benchmark results

Device `cuda`, requested dtype `float32`.

| Example | Variant | Inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | Peak MB |
|---|---|---:|---:|---:|---:|---:|
| stable_var_p32_t256 | exact_graph_only | 0.254 | 0.99x | 1.0000 | 0 | 1.8 |
| stable_var_p32_t256 | downsample_lowrank_exact_h | 0.266 | 0.95x | 0.6865 | 0 | 2.1 |
| stable_var_p32_t256 | downsample_lowrank_h1_r16 | 0.261 | 0.97x | 0.1434 | 0.072209 | 2.1 |
| stable_var_p32_t256 | downsample_lowrank_learned_h_r4 | 0.287 | 0.88x | 0.0072 | 1.0129 | 2.1 |
