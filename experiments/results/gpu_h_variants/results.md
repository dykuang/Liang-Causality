# LK benchmark results

Device `cuda`, requested dtype `float32`.

| Example | Variant | Inference (ms) | H inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | Peak MB |
|---|---|---:|---:|---:|---:|---:|---:|
| sparse_var_p128_t512 | exact_graph_only | 1.422 | 0.197 | 0.99x | 1.0000 | 0 | 59.1 |
| sparse_var_p128_t512 | exact_h | 0.525 | 0.203 | 2.69x | 0.4864 | 0 | 71.6 |
| sparse_var_p128_t512 | proj_h8 | 0.486 | 0.163 | 2.90x | 0.0293 | 0.16636 | 71.6 |
| sparse_var_p128_t512 | proj_h16 | 0.487 | 0.164 | 2.89x | 0.0362 | 0.071885 | 71.6 |
| sparse_var_p128_t512 | proj_h32 | 0.489 | 0.166 | 2.88x | 0.0421 | 0.033316 | 71.7 |
| sparse_var_p128_t512 | proj_h64 | 0.493 | 0.175 | 2.86x | 0.0592 | 0.0163 | 71.7 |
| sparse_var_p128_t512 | pair_h | 6.438 | 6.377 | 0.22x | 0.0331 | 0.0094887 | 623.6 |
| sparse_var_p128_t512 | lr_h4 | 0.396 | 0.341 | 3.56x | 0.0326 | 0.25077 | 56.2 |
| sparse_var_p128_t512 | lr_h8 | 0.402 | 0.339 | 3.51x | 0.0348 | 0.12539 | 56.2 |
| sparse_var_p128_t512 | lr_h16 | 0.399 | 0.343 | 3.53x | 0.0359 | 0.11157 | 56.2 |
| sparse_var_p128_t512 | lr_h32 | 0.413 | 0.352 | 3.42x | 0.0364 | 0.11132 | 56.2 |
