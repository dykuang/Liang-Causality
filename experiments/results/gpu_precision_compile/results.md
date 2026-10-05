# LK benchmark results

Device `cuda`, requested dtype `bfloat16`.

| Example | Variant | Inference (ms) | H inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | ρ symmetry err. | ρ diagonal err. | Peak MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_var_p128_t512 | exact_graph_only | 1.148 | 0.217 | 1.00x | 1.0000 | 0 | NA | NA | 66.3 |
| sparse_var_p128_t512 | exact_h_lrW16 | 0.254 | 0.156 | 4.52x | 0.4870 | 0 | NA | NA | 271.2 |
| sparse_var_p128_t512 | lr_h8_lrW16 | 0.134 | 0.119 | 8.56x | 0.0329 | 0.1278 | 0 | 0 | 294.2 |
