# LK benchmark results

Device `cuda`, requested dtype `float32`.

| Example | Variant | Inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | Peak MB |
|---|---|---:|---:|---:|---:|---:|
| sparse_var_p128_t512 | exact_graph_only | 1.313 | 1.00x | 1.0000 | 0 | 60.6 |
| sparse_var_p128_t512 | exact_h | 0.512 | 2.57x | 0.4874 | 0 | 55.1 |
| sparse_var_p128_t512 | proj_h8 | 0.478 | 2.75x | 0.0306 | 0.16636 | 55.2 |
| sparse_var_p128_t512 | proj_h32 | 0.481 | 2.73x | 0.0440 | 0.033316 | 55.2 |
| sparse_var_p128_t512 | pair_h | 3.423 | 0.38x | 0.0314 | 0.009453 | 607.2 |
| sparse_var_p128_t512 | lr_h4 | 0.410 | 3.20x | 0.0344 | 0.25353 | 39.7 |
| sparse_var_p128_t512 | lr_h8 | 0.397 | 3.31x | 0.0313 | 0.12647 | 39.7 |
| sparse_var_p128_t512 | lr_h16 | 0.397 | 3.31x | 0.0355 | 0.10129 | 39.7 |
