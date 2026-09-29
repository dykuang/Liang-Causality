# Causality Inference via Liang-Kleeman information flow  
 A repo for implementing causality matrix with the method of Liang's information flow.

* `causality_estimation.py` contains the implementation using `numpy`.    
* `Causality_est_torch.py` contains the implementation using `torch`, where the method can be accessed via a regular function interface or as a torch module that can be directly integrated in neural networks' design as a layer.    
* `amortized_lk.py` contains a label-free, permutation-equivariant LK approximation trained with matrix-free stochastic normal-equation residuals. It does not compute exact LK labels or solve the LK system.

## Label-free amortized LK operator

`AmortizedLK` predicts the regression coefficient matrix from each time-series window and combines it with a random-projection estimate of the covariance-ratio factor. Inputs have shape `(batch, variables, time)`; the module returns `(causal_matrix, coefficients, covariance_ratio)`.

```python
from amortized_lk import AmortizedLK, stochastic_probe_residual_loss

model = AmortizedLK(width=32, projection_rank=64, n_step=1, dt=1.0)
causal_hat, coefficients_hat, ratio_hat = model(windows)
loss = stochastic_probe_residual_loss(
    windows, coefficients_hat, q=4, n_step=1, dt=1.0
)
loss.backward()
```

The Rademacher probe loss is an unbiased estimate of the ridge normal-equation residual objective when `normalized=False` (the default). Pass a fixed temporal `projection` to `AmortizedLK.forward` to make the random-projection covariance-ratio estimate reproducible for a fixed window length. Exact LK may be used separately after training for evaluation, but it is not part of this training objective.

For a no-teacher synthetic run, use `python train_amortized_lk.py --steps 1000 --output amortized_lk.pt`. Vary dimensions during training with `--variables-min 4 --variables-max 16 --time-min 128 --time-max 512`. Add `--h-method learned` to train the H2 covariance-ratio head from sampled variable pairs; otherwise H1 uses the temporal projection. Evaluate a saved checkpoint separately with `python evaluate_amortized_lk.py amortized_lk.pt`. Run the compact probe-count, probe-distribution, ridge, and H1-rank study with `python validate_amortized_lk.py`.

### Selecting exact LK or the learned operator

`LKEstimator` gives both paths one interface. The exact path returns the graph, variance, and normalized graph; the learned path returns the graph, coefficients, and covariance ratio (variance is not estimated).

```python
from lk_estimators import LKEstimator, load_amortized_checkpoint

exact = LKEstimator(mode="exact", n_step=1, dt=1.0)
exact_result = exact(windows)

model = load_amortized_checkpoint("amortized_lk.pt")
learned = LKEstimator(mode="amortized", amortized_model=model)
learned_result = learned(windows)
```

The learned architecture is configurable with `encoder_type="tcn" | "downsample_mlp" | "spectral" | "attention"`, `decoder_type="pair_mlp" | "bilinear" | "low_rank" | "bilinear_residual"`, and `h_method="projection" | "learned_pair" | "learned_low_rank" | "exact"` (`learned` remains an alias for `learned_pair`). Exact H is a regular label-free mode: it computes the covariance-ratio factor but never an LK coefficient solve. Use `compute_graph=False` when coefficient-residual training does not need a graph.

### CPU ablation

Recreate the run with `python -m pip install -r requirements-amortized.txt` followed by `python benchmark_lk_ablations.py experiments/ablation_cpu.yaml`. The YAML file defines the data examples, model variants, training steps, and timing settings. Learned coefficient models train from probe residuals; H2 adds sampled covariance-ratio supervision. Exact LK is run after training to score the learned graphs. Full exact timing includes the legacy variance calculation, while `exact_graph_only` measures exact graph estimation without variance. Learned outputs include the graph, coefficients, and covariance ratio, but no variance estimate.

Results below used PyTorch 2.11.0+cpu with one CPU thread, 150 training steps for each learned setting (80 for the p=32 case), batch size 8 (4 for p=32), and 20 timing repetitions. Inference milliseconds and training seconds are batch timings. These are small exploratory runs, not a speedup claim: this CPU and these dimensions generally favor exact LK, while training also has a nontrivial one-time cost. Graph MSE is against exact LK; rank correlation and sign accuracy are calculated across edges.

| Example | Variant | Train (s) | Inference (ms) | Speedup | Graph MSE | Magnitude rank corr. | Strong sign acc. |
|---|---|---:|---:|---:|---:|---:|---:|
| small_oscillator_p2_t256 (p=2, T=256) | exact_lk | 0.00 | 0.931 | 1.00x | 0 | 1.0000 | 1.0000 |
| small_oscillator_p2_t256 (p=2, T=256) | exact_graph_only | 0.00 | 0.292 | 3.19x | 4.967e-18 | 1.0000 | 1.0000 |
| small_oscillator_p2_t256 (p=2, T=256) | tcn_pair_h1 | 2.51 | 1.709 | 0.54x | 0.00144123 | 0.5755 | 0.8125 |
| small_oscillator_p2_t256 (p=2, T=256) | tcn_pair_h0_exact_H | 2.68 | 1.743 | 0.53x | 0.00144424 | 0.6745 | 0.8125 |
| small_oscillator_p2_t256 (p=2, T=256) | tcn_pair_h2 | 2.38 | 1.623 | 0.57x | 0.00175193 | 0.2489 | 0.9375 |
| small_oscillator_p2_t256 (p=2, T=256) | downsample_bilinear_h1 | 1.71 | 0.827 | 1.13x | 0.00125101 | 0.5946 | 0.9375 |
| small_oscillator_p2_t256 (p=2, T=256) | spectral_lowrank_h1 | 1.75 | 0.769 | 1.21x | 0.00127972 | 0.3402 | 0.8750 |
| small_oscillator_p2_t256 (p=2, T=256) | attention_bilinear_residual_h1 | 3.09 | 3.821 | 0.24x | 0.00188326 | 0.4326 | 0.6875 |
| repository_6d_t512 (p=6, T=512) | exact_lk | 0.00 | 1.213 | 1.00x | 0 | 1.0000 | 1.0000 |
| repository_6d_t512 (p=6, T=512) | exact_graph_only | 0.00 | 0.378 | 3.21x | 1.69478e-16 | 1.0000 | 1.0000 |
| repository_6d_t512 (p=6, T=512) | tcn_pair_h1 | 4.74 | 4.020 | 0.30x | 0.102291 | 0.6258 | 0.6944 |
| repository_6d_t512 (p=6, T=512) | tcn_pair_h0_exact_H | 4.61 | 5.060 | 0.24x | 0.0962113 | 0.7451 | 0.6597 |
| repository_6d_t512 (p=6, T=512) | tcn_pair_h2 | 4.74 | 4.928 | 0.25x | 0.112704 | 0.4284 | 0.6528 |
| repository_6d_t512 (p=6, T=512) | downsample_bilinear_h1 | 3.05 | 1.454 | 0.83x | 0.113624 | 0.5914 | 0.6875 |
| repository_6d_t512 (p=6, T=512) | spectral_lowrank_h1 | 3.09 | 1.527 | 0.79x | 0.124962 | 0.5418 | 0.6875 |
| repository_6d_t512 (p=6, T=512) | attention_bilinear_residual_h1 | 6.60 | 12.400 | 0.10x | 0.113767 | 0.5642 | 0.6528 |
| sparse_var_p32_t512 (p=32, T=512) | exact_lk | 0.00 | 1.586 | 1.00x | 0 | 1.0000 | 1.0000 |
| sparse_var_p32_t512 (p=32, T=512) | exact_graph_only | 0.00 | 0.705 | 2.25x | 4.40777e-17 | 1.0000 | 1.0000 |
| sparse_var_p32_t512 (p=32, T=512) | tcn_pair_h1 | 3.91 | 12.019 | 0.13x | 0.0323748 | 0.2266 | 0.5269 |
| sparse_var_p32_t512 (p=32, T=512) | tcn_pair_h0_exact_H | 4.30 | 12.926 | 0.12x | 0.0323741 | 0.7006 | 0.5244 |
| sparse_var_p32_t512 (p=32, T=512) | tcn_pair_h2 | 4.23 | 12.474 | 0.13x | 0.0343556 | -0.0757 | 0.5137 |
| sparse_var_p32_t512 (p=32, T=512) | downsample_bilinear_h1 | 1.52 | 1.703 | 0.93x | 0.0323034 | 0.2202 | 0.5269 |
| sparse_var_p32_t512 (p=32, T=512) | spectral_lowrank_h1 | 1.63 | 1.691 | 0.94x | 0.0319889 | 0.2241 | 0.5269 |
| sparse_var_p32_t512 (p=32, T=512) | attention_bilinear_residual_h1 | 6.36 | 26.906 | 0.06x | 0.0346577 | 0.1384 | 0.4878 |

The machine-readable results and copied run config are in `experiments/results/cpu_ablation/`.

### GPU H variants and crossover benchmarks

`AmortizedLK` also supports `h_method="exact"` during training (it computes only the covariance ratio, not an LK solve), `h_method="learned_pair"` (the compatibility name for `learned`), and `h_method="learned_low_rank"`. The latter constructs a symmetric PSD low-rank correlation matrix and combines it with exact per-channel RMS ratios. Set `compute_graph=False` to train coefficients without materializing H or the graph; `score_h_pairs()` supports sampled low-rank-H supervision without a dense H allocation.

GPU suites remain YAML-driven and are run through the same entry point:

- `env -u LD_LIBRARY_PATH python benchmark_lk_ablations.py experiments/gpu_smoke.yaml`
- `env -u LD_LIBRARY_PATH python benchmark_lk_ablations.py experiments/gpu_crossover.yaml`
- `env -u LD_LIBRARY_PATH python benchmark_lk_ablations.py experiments/gpu_h_variants.yaml`
- `env -u LD_LIBRARY_PATH python benchmark_lk_ablations.py experiments/gpu_precision_compile.yaml`

Each output includes the copied configuration, machine-readable CSV, and Markdown summary. CUDA runs use warmup iterations, CUDA-event latency, synchronized wall latency, peak allocated memory, requested/effective dtype, compile metadata, H-isolation metrics, and speedup against graph-only exact LK. The `env -u LD_LIBRARY_PATH` prefix is needed on this machine because its CUDA 11.2 system library path conflicts with PyTorch's bundled CUDA 11.7 libraries; it is not normally necessary.

## GPU benchmark results

Benchmarks were run on an NVIDIA GeForce RTX 3080 Ti (12 GB), PyTorch 1.13.1+cu117, with CUDA-event latency after warmup. All amortized models used the downsampled-MLP encoder, a rank-16 coefficient decoder, label-free probe-residual training, and batch size 32. The exact baseline is graph-only exact LK; it is the relevant comparison because amortized inference does not return exact-LK variance.

### FP32 crossover: T=512

| Variables | Exact graph-only (ms) | Exact-H learned-W (ms / speedup / rank corr.) | Low-rank-H r=8 (ms / speedup / rank corr.) |
|---:|---:|---:|---:|
| 32 | 0.347 | 0.295 / 1.18x / 0.5628 | 0.312 / 1.12x / 0.1099 |
| 64 | 0.677 | 0.337 / 2.01x / 0.5158 | 0.301 / 2.25x / 0.0700 |
| 128 | 1.299 | 0.511 / 2.54x / 0.4887 | 0.393 / 3.30x / 0.0339 |
| 256 | 3.126 | 1.001 / 3.13x / 0.4819 | 0.843 / 3.72x / 0.0206 |

Exact-H provides the useful fidelity/speed trade-off in this initial sweep: it reaches 2--3x graph-only-exact speedup for p >= 64 while retaining roughly 0.48--0.56 graph magnitude-rank correlation. Low-rank H is faster and uses less peak memory, but its final graph rank correlation is currently poor; it should be treated as a speed/memory ablation, not the default fidelity option.

### H-head isolation: FP32, p=128, T=512

| H method | Latency (ms) | Speedup | H MSE | Graph rank corr. | Peak memory (MB) |
|---|---:|---:|---:|---:|---:|
| Exact H | 0.512 | 2.57x | 0 | 0.4874 | 55.1 |
| Projection H (r=32) | 0.481 | 2.73x | 0.0333 | 0.0440 | 55.2 |
| Learned pair H | 3.423 | 0.38x | 0.00945 | 0.0314 | 607.2 |
| Learned low-rank H (r=8) | 0.397 | 3.31x | 0.1265 | 0.0313 | 39.7 |

### BF16 eager: p=128, T=512

| Variant | Latency (ms) | Speedup | Graph rank corr. | Peak memory (MB) |
|---|---:|---:|---:|---:|
| Exact graph-only (FP32 effective) | 1.295 | 1.00x | 1.0000 | 61.1 |
| Exact-H learned-W | 0.333 | 3.88x | 0.4906 | 35.3 |
| Learned low-rank H (r=8) | 0.313 | 4.13x | 0.0279 | 28.1 |

Raw CSV/Markdown outputs and copied configurations are in `experiments/results/gpu_crossover/`, `experiments/results/gpu_h_variants/`, `experiments/results/gpu_precision_eager/`, and `experiments/results/gpu_smoke/`. The `gpu_precision_compile.yaml` suite fails early and explicitly on this host because PyTorch 1.13.1 does not provide `torch.compile`; eager BF16 is reported above instead.

 ## EXAMPLE USAGE  
Suppose we have observed two time series as:  

![ts](https://github.com/user-attachments/assets/572fcffe-96ed-4c65-a26f-eda1ab22ce6c)  

We now want to query the possible causal relationship between $x$ and $y$.  

Suppose the data is collected in variable $xy$ (a 1001 by 2 data matrix observed with time step $\Delta t = 0.1$). Run  
```
cau, var, cau_normalized = causal_est_matrix(xy.T, n_step=1, dt=0.1)
```
to obtain  
* causality matrix `cau` whose $ij$'s entry indicates causality from $i$ to $j$ (the value can be positive and negative, positive values can be interpreted as "encouragement" effect while negative values can be interpreted as "surpression" effect),   
* variance matrix for the estimation `var` whose $ij$'s entry is the variance for estimated causality value of $ij$'s entry in `cau` and normalized causality matrix `cau_normalized`.

Here `causal_est_matrix` is from `causality_estimation.py`, `n_step` gives the steps user specified for Euler finite difference scheme for estimation of derivatives in the internal algorithm. A further statistical test as below:
```
np.abs(cau)>np.sqrt(var)*2.56
```
provides the causality matrix (about 98.96% $\approx$ 99% significant level for about 2.56 standard deviation away from 0) as a boolean type:
```
False, False  
True, True
```
It actually reveals the true dynamics under the observation since the data is generated by the stochastic system ($\epsilon_1$ and $\epsilon_1$ are 1d Brownian motions):  
$\frac{dx}{dt} = y + \epsilon_1; \frac{dy}{dt} = -y + \epsilon_2$.   

Why? Because the first equation indicates the dynamic of $x$ is internally driven by $y$ alone ($y\rightarrow x$) and the second equation tells that $y$ is driven by itself ($y\rightarrow y$). Above causality is naturally: 

False ($x\rightarrow x$), False ($x\rightarrow y$)  
True ($y\rightarrow y$), True ($y\rightarrow x$)  

As a simple summary for linear ODE as $\frac{dX}{dt} = AX + \epsilon$, if $A_{ij} \neq 0$, then you should expect the boolean type causality matrix $C^B$ with $C_{ji}^B$ equals `True`. You can check this with the above example where $A$ is 
```
 0  1
 0 -1
```

You can use the following code to generate your own toy data:  
```
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
xy = np.zeros((t_eval.shape[0]+1,2))
xy[0] = np.array([0,1]).T
dt = 0.1
sigma = 0.01

for i in range(t_eval.shape[0]):
    xy[i+1][0] = xy[i][1] * dt + xy[i][0] + sigma*np.random.randn()
    xy[i+1][1] = -xy[i][1] * dt + xy[i][1] + sigma*np.random.randn()
```

If you would like to use a torch implementation, you can use function `causal_est_matrix` from `Causality_est_torch.py` as 
```
XY = torch.tensor(xy.T, dtype=torch.float32)
causality,variance,normalized_causality = causal_est_matrix(XY)
```  
or torch module `LiangCausalityEstimator` from the same file as:  
```
XY = torch.tensor(xy.T, dtype=torch.float32) 
model = LiangCausalityEstimator(n_step=1, dt=1)
XY = XY.unsqueeze(0)  # Add batch dimension
causality, variance, normalized_causality = model(XY)
```
They should provide the same result (`float32` works for most of the time, but `float64` may be needed for some cases). Please find some more examples in `example.ipynb`.


## Reference
Please kindly cite this repository and following papers if you find this repo helpful in your project.  
    📑 X.San Liang, "[Normalized Multivariate Time Series Causality Analysis and Causal Graph Reconstruction](https://www.mdpi.com/1099-4300/23/6/679)", (2021)   
    📑 X.San Liang, "[Information flow and causality as rigorous notions ab initio](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.052201)",(2016)  
    📑 X.San Liang, "[Normalizing the causality between time series](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.92.022126)",(2015)  
    📑 X.San Liang, "[Unraveling the cause-effect relation between time series](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.90.052150)",(2014)  
    📑 X.San Liang, "[The Liang-Kleeman Information Flow: Theory and Applications](https://www.mdpi.com/1099-4300/15/1/327)",(2013)  
    📑 X.San Liang, Richard Kleeman, "[Information Transfer between Dynamical System Components](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.95.244101)",(2005)
    
