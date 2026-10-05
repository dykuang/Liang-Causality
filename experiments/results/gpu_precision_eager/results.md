# LK benchmark results

Device `cuda`, requested dtype `bfloat16`.

| Example | Variant | Inference (ms) | H inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | ρ symmetry err. | ρ diagonal err. | Peak MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_var_p128_t512 | exact_graph_only | 1.482 | 0.214 | 1.01x | 1.0000 | 0 | NA | NA | 66.3 |
| sparse_var_p128_t512 | exact_h_lrW16 | 0.415 | 0.215 | 3.60x | 0.4871 | 0 | NA | NA | 76.6 |
| sparse_var_p128_t512 | lr_h8_lrW16 | 0.405 | 0.225 | 3.69x | 0.0324 | 0.1279 | 0 | 0 | 49.8 |
