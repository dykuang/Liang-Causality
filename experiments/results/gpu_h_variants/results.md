# LK benchmark results

Device `cuda`, requested dtype `float32`.

| Example | Variant | Inference (ms) | H inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | ρ symmetry err. | ρ diagonal err. | Peak MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_var_p128_t512 | exact_graph_only | 1.453 | 0.202 | 1.01x | 1.0000 | 0 | NA | NA | 59.1 |
| sparse_var_p128_t512 | exact_h | 0.520 | 0.202 | 2.83x | 0.4864 | 0 | NA | NA | 71.6 |
| sparse_var_p128_t512 | proj_h8 | 0.480 | 0.157 | 3.07x | 0.0293 | 0.16636 | NA | NA | 71.6 |
| sparse_var_p128_t512 | proj_h16 | 0.481 | 0.164 | 3.06x | 0.0362 | 0.071885 | NA | NA | 71.6 |
| sparse_var_p128_t512 | proj_h32 | 0.483 | 0.160 | 3.05x | 0.0421 | 0.033316 | NA | NA | 71.7 |
| sparse_var_p128_t512 | proj_h64 | 0.493 | 0.166 | 2.99x | 0.0592 | 0.0163 | NA | NA | 71.7 |
| sparse_var_p128_t512 | pair_h | 6.645 | 6.468 | 0.22x | 0.0331 | 0.0094887 | NA | NA | 623.6 |
| sparse_var_p128_t512 | lr_h4 | 0.394 | 0.338 | 3.73x | 0.0326 | 0.25077 | 0 | 0 | 58.2 |
| sparse_var_p128_t512 | lr_h8 | 0.400 | 0.341 | 3.68x | 0.0348 | 0.12539 | 0 | 0 | 58.2 |
| sparse_var_p128_t512 | lr_h16 | 0.397 | 0.341 | 3.71x | 0.0359 | 0.11157 | 0 | 0 | 58.2 |
| sparse_var_p128_t512 | lr_h32 | 0.403 | 0.347 | 3.65x | 0.0364 | 0.11132 | 0 | 0 | 58.2 |
