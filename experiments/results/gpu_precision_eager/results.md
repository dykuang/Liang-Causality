# LK benchmark results

Device `cuda`, requested dtype `bfloat16`.

| Example | Variant | Inference (ms) | H inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | ρ symmetry err. | ρ diagonal err. | Peak MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_var_p128_t512 | exact_graph_only | 1.145 | 0.216 | 1.00x | 1.0000 | 0 | NA | NA | 66.3 |
| sparse_var_p128_t512 | exact_h_lrW16 | 0.518 | 0.217 | 2.21x | 0.4870 | 0 | NA | NA | 80.6 |
| sparse_var_p128_t512 | lr_h8_lrW16 | 0.398 | 0.324 | 2.88x | 0.0329 | 0.1278 | 0 | 0 | 66.4 |
