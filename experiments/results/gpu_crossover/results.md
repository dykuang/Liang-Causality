# LK benchmark results

Device `cuda`, requested dtype `float32`.

| Example | Variant | Inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | Peak MB |
|---|---|---:|---:|---:|---:|---:|
| sparse_var_p32_t512 | exact_graph_only | 0.347 | 1.00x | 1.0000 | 0 | 13.2 |
| sparse_var_p32_t512 | downsample_lrW16_exactH | 0.295 | 1.18x | 0.5628 | 0 | 11.5 |
| sparse_var_p32_t512 | downsample_lrW16_projH16 | 0.283 | 1.23x | 0.1265 | 0.077273 | 11.5 |
| sparse_var_p32_t512 | downsample_lrW16_projH32 | 0.272 | 1.28x | 0.1593 | 0.034814 | 11.5 |
| sparse_var_p32_t512 | downsample_lrW16_lrH4 | 0.298 | 1.17x | 0.1084 | 0.27248 | 8.7 |
| sparse_var_p32_t512 | downsample_lrW16_lrH8 | 0.312 | 1.12x | 0.1099 | 0.14682 | 8.7 |
| sparse_var_p32_t512 | downsample_lrW16_lrH16 | 0.308 | 1.13x | 0.1223 | 0.092166 | 8.7 |
| sparse_var_p64_t512 | exact_graph_only | 0.677 | 1.00x | 1.0000 | 0 | 27.7 |
| sparse_var_p64_t512 | downsample_lrW16_exactH | 0.337 | 2.01x | 0.5158 | 0 | 24.2 |
| sparse_var_p64_t512 | downsample_lrW16_projH16 | 0.301 | 2.26x | 0.0752 | 0.0722 | 24.2 |
| sparse_var_p64_t512 | downsample_lrW16_projH32 | 0.306 | 2.22x | 0.0810 | 0.033589 | 24.3 |
| sparse_var_p64_t512 | downsample_lrW16_lrH4 | 0.289 | 2.35x | 0.0573 | 0.25442 | 17.5 |
| sparse_var_p64_t512 | downsample_lrW16_lrH8 | 0.301 | 2.25x | 0.0700 | 0.13159 | 17.5 |
| sparse_var_p64_t512 | downsample_lrW16_lrH16 | 0.294 | 2.31x | 0.0683 | 0.11619 | 17.5 |
| sparse_var_p128_t512 | exact_graph_only | 1.299 | 1.00x | 1.0000 | 0 | 61.0 |
| sparse_var_p128_t512 | downsample_lrW16_exactH | 0.511 | 2.54x | 0.4887 | 0 | 55.0 |
| sparse_var_p128_t512 | downsample_lrW16_projH16 | 0.479 | 2.71x | 0.0410 | 0.071391 | 55.0 |
| sparse_var_p128_t512 | downsample_lrW16_projH32 | 0.482 | 2.69x | 0.0483 | 0.033532 | 55.0 |
| sparse_var_p128_t512 | downsample_lrW16_lrH4 | 0.393 | 3.30x | 0.0324 | 0.25108 | 39.5 |
| sparse_var_p128_t512 | downsample_lrW16_lrH8 | 0.393 | 3.30x | 0.0339 | 0.12584 | 39.5 |
| sparse_var_p128_t512 | downsample_lrW16_lrH16 | 0.396 | 3.28x | 0.0398 | 0.083844 | 39.5 |
| sparse_var_p256_t512 | exact_graph_only | 3.126 | 1.00x | 1.0000 | 0 | 146.6 |
| sparse_var_p256_t512 | downsample_lrW16_exactH | 1.001 | 3.13x | 0.4819 | 0 | 137.5 |
| sparse_var_p256_t512 | downsample_lrW16_projH16 | 0.935 | 3.35x | 0.0260 | 0.071558 | 138.0 |
| sparse_var_p256_t512 | downsample_lrW16_projH32 | 0.937 | 3.34x | 0.0313 | 0.033281 | 138.6 |
| sparse_var_p256_t512 | downsample_lrW16_lrH4 | 0.845 | 3.71x | 0.0203 | 0.24958 | 121.7 |
| sparse_var_p256_t512 | downsample_lrW16_lrH8 | 0.843 | 3.72x | 0.0206 | 0.12499 | 121.8 |
| sparse_var_p256_t512 | downsample_lrW16_lrH16 | 0.850 | 3.68x | 0.0223 | 0.090351 | 122.0 |
