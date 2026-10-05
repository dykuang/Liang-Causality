# LK benchmark results

Device `cuda`, requested dtype `bfloat16`.

| Example | Variant | Inference (ms) | H inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | ρ symmetry err. | ρ diagonal err. | Peak MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_var_p128_t512 | exact_graph_only | 1.480 | 0.217 | 1.04x | 1.0000 | 0 | NA | NA | 66.3 |
| sparse_var_p128_t512 | exact_h_lrW16 | 0.267 | 0.165 | 5.75x | 0.4871 | 0 | NA | NA | 50.5 |
| sparse_var_p128_t512 | lr_h8_lrW16 | 0.135 | 0.119 | 11.37x | 0.0324 | 0.1279 | 0 | 0 | 281.4 |
