# LK benchmark results

Device `cuda`, requested dtype `float32`.

| Example | Variant | Inference (ms) | H inference (ms) | Speedup vs graph-only exact | Graph rank corr. | H MSE | ρ symmetry err. | ρ diagonal err. | Peak MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_var_p32_t512 | exact_graph_only | 0.315 | 0.110 | 1.02x | 1.0000 | 0 | NA | NA | 25.1 |
| sparse_var_p32_t512 | downsample_lrW16_exactH | 0.376 | 0.113 | 0.86x | 0.5828 | 0 | NA | NA | 27.7 |
| sparse_var_p32_t512 | downsample_lrW16_projH16 | 0.353 | 0.074 | 0.91x | 0.1279 | 0.077273 | NA | NA | 27.8 |
| sparse_var_p32_t512 | downsample_lrW16_projH32 | 0.339 | 0.072 | 0.95x | 0.1521 | 0.034814 | NA | NA | 27.8 |
| sparse_var_p32_t512 | downsample_lrW16_lrH4 | 0.352 | 0.271 | 0.91x | 0.1080 | 0.26509 | 0 | 0 | 25.1 |
| sparse_var_p32_t512 | downsample_lrW16_lrH8 | 0.364 | 0.281 | 0.89x | 0.1132 | 0.14086 | 0 | 0 | 25.1 |
| sparse_var_p32_t512 | downsample_lrW16_lrH16 | 0.363 | 0.279 | 0.89x | 0.1240 | 0.096056 | 0 | 0 | 25.1 |
| sparse_var_p64_t512 | exact_graph_only | 0.920 | 0.135 | 0.94x | 1.0000 | 0 | NA | NA | 34.4 |
| sparse_var_p64_t512 | downsample_lrW16_exactH | 0.315 | 0.135 | 2.74x | 0.5170 | 0 | NA | NA | 40.5 |
| sparse_var_p64_t512 | downsample_lrW16_projH16 | 0.274 | 0.093 | 3.15x | 0.0730 | 0.0722 | NA | NA | 40.5 |
| sparse_var_p64_t512 | downsample_lrW16_projH32 | 0.275 | 0.095 | 3.14x | 0.0840 | 0.033589 | NA | NA | 40.5 |
| sparse_var_p64_t512 | downsample_lrW16_lrH4 | 0.300 | 0.268 | 2.88x | 0.0633 | 0.25569 | 0 | 0 | 34.2 |
| sparse_var_p64_t512 | downsample_lrW16_lrH8 | 0.358 | 0.267 | 2.41x | 0.0686 | 0.1298 | 0 | 0 | 34.2 |
| sparse_var_p64_t512 | downsample_lrW16_lrH16 | 0.371 | 0.279 | 2.33x | 0.0718 | 0.1043 | 0 | 0 | 34.2 |
| sparse_var_p128_t512 | exact_graph_only | 1.511 | 0.202 | 0.99x | 1.0000 | 0 | NA | NA | 59.3 |
| sparse_var_p128_t512 | downsample_lrW16_exactH | 0.519 | 0.196 | 2.87x | 0.4899 | 0 | NA | NA | 71.2 |
| sparse_var_p128_t512 | downsample_lrW16_projH16 | 0.480 | 0.163 | 3.10x | 0.0393 | 0.071391 | NA | NA | 71.3 |
| sparse_var_p128_t512 | downsample_lrW16_projH32 | 0.482 | 0.165 | 3.09x | 0.0471 | 0.033532 | NA | NA | 71.3 |
| sparse_var_p128_t512 | downsample_lrW16_lrH4 | 0.393 | 0.337 | 3.79x | 0.0371 | 0.25035 | 0 | 0 | 57.8 |
| sparse_var_p128_t512 | downsample_lrW16_lrH8 | 0.401 | 0.341 | 3.72x | 0.0372 | 0.12595 | 0 | 0 | 57.8 |
| sparse_var_p128_t512 | downsample_lrW16_lrH16 | 0.401 | 0.341 | 3.71x | 0.0391 | 0.084118 | 0 | 0 | 57.8 |
| sparse_var_p256_t512 | exact_graph_only | 3.063 | 0.411 | 1.09x | 1.0000 | 0 | NA | NA | 138.9 |
| sparse_var_p256_t512 | downsample_lrW16_exactH | 1.044 | 0.411 | 3.20x | 0.4810 | 0 | NA | NA | 161.8 |
| sparse_var_p256_t512 | downsample_lrW16_projH16 | 0.934 | 0.301 | 3.58x | 0.0259 | 0.071558 | NA | NA | 154.3 |
| sparse_var_p256_t512 | downsample_lrW16_projH32 | 0.936 | 0.303 | 3.57x | 0.0321 | 0.033281 | NA | NA | 154.8 |
| sparse_var_p256_t512 | downsample_lrW16_lrH4 | 0.842 | 0.684 | 3.97x | 0.0199 | 0.25008 | 0 | 0 | 137.9 |
| sparse_var_p256_t512 | downsample_lrW16_lrH8 | 0.841 | 0.683 | 3.97x | 0.0202 | 0.12485 | 0 | 0 | 138.0 |
| sparse_var_p256_t512 | downsample_lrW16_lrH16 | 0.848 | 0.690 | 3.94x | 0.0228 | 0.083031 | 0 | 0 | 138.3 |
