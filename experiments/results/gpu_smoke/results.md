# LK benchmark results

Device `cuda`, requested dtype `float32`.

| Example | Variant | Inference (ms) | H inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | ρ symmetry err. | ρ diagonal err. | Peak MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stable_var_p32_t256 | exact_graph_only | 0.233 | 0.083 | 1.56x | 1.0000 | 0 | NA | NA | 17.4 |
| stable_var_p32_t256 | downsample_lowrank_exact_h | 0.361 | 0.070 | 1.01x | 0.6867 | 0 | NA | NA | 18.4 |
| stable_var_p32_t256 | downsample_lowrank_h1_r16 | 0.133 | 0.072 | 2.74x | 0.1446 | 0.072209 | NA | NA | 18.4 |
| stable_var_p32_t256 | downsample_lowrank_learned_h_r4 | 0.360 | 0.226 | 1.02x | 0.0821 | 0.45499 | 0 | 0 | 18.4 |
