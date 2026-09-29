# LK benchmark results

Device `cuda`, requested dtype `bfloat16`.

| Example | Variant | Inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | Peak MB |
|---|---|---:|---:|---:|---:|---:|
| sparse_var_p128_t512 | exact_graph_only | 1.295 | 1.00x | 1.0000 | 0 | 61.1 |
| sparse_var_p128_t512 | exact_h_lrW16 | 0.333 | 3.88x | 0.4906 | 0 | 35.3 |
| sparse_var_p128_t512 | lr_h8_lrW16 | 0.313 | 4.13x | 0.0279 | 0.0099495 | 28.1 |
