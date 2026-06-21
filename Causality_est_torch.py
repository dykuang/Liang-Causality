'''
author @ Dongyang Kuang
Functions for calculating the LK information flow based causality

Reference:
    X.San Liang, "Normalized Multivariate Time Series Causality Analysis and
    Causal Graph Reconstruction", (2021)

Conditions for the LK information flow based causality:
   : stationarity
   : linearity
   : additive noises

but proved to be effective in many cases when condtions listed above are 
not fully satisfied.

This version is a PyTorch implementation with GPU optimizations:

GPU Optimizations:
    - Vectorized batch processing in LiangCausalityEstimator (process many time
      series at once; ~100x faster per series than looping single calls).
    - Variance estimation via a Schur-complement / Woodbury reformulation of the
      Fisher information matrix (_variance_diag): instead of building and
      inverting B*nx bordered (nx+2) x (nx+2) matrices -- a (B, nx, nx+2, nx+2)
      tensor that is the dominant cost/memory for large batches -- it inverts
      only one nx x nx matrix per batch item plus closed-form 2x2 inverses.
      Results match the bordered-inverse approach to machine precision.
    - Batched matrix operations (bmm/einsum) throughout; explicit device control.

Performance Tips:
    - Process multiple time series in batches via LiangCausalityEstimator rather
      than calling causal_est_matrix in a Python loop.
    - Keep data on GPU to minimize CPU-GPU transfers.
    - Use float64 for ill-conditioned covariances; float32 is faster when stable.
    - torch.compile() in PyTorch 2.0+ can fuse the many small ops for extra speed.
'''
#%%
import numpy as np
import torch


def _variance_diag(C, X_slice, RR, bb, dt, N):
    """Variance of the causality estimates via the Fisher information matrix.

    For every (batch item, target variable i) Liang's estimator forms an
    ``(nx+2) x (nx+2)`` Fisher information matrix and needs the diagonal of its
    inverse over the ``nx`` regression-coefficient indices. The matrix is
    *bordered*: an ``nx x nx`` middle block ``(dt/b_i^2) G`` (where
    ``G = X_slice X_slice^T`` is shared across i) plus two border rows/cols.

    Instead of building and inverting ``B*nx`` matrices of size ``nx+2`` (which
    for a large batch allocates a ``(B, nx, nx+2, nx+2)`` tensor), we use the
    Schur complement on the 2 border indices and the Woodbury identity: only one
    ``nx x nx`` inverse per item (of the shared ``G``) plus closed-form ``2x2``
    inverses. Results are identical to the bordered-inverse approach to machine
    precision.

    Args (all with a leading batch dim ``B``):
        C:       (B, nx, nx)  covariance of X_slice
        X_slice: (B, nx, T)   the regressor history X[:, :-n_step]
        RR:      (B, nx, T)    residuals
        bb:      (B, nx)       per-variable noise amplitude
        dt, N:   scalars
    Returns:
        var: (B, nx, nx) with var[b, i, j] the variance for target i, regressor j
        (the caller transposes the last two dims to match the public layout).
    """
    alpha = dt / bb**2                                       # (B, nx)
    G = X_slice @ X_slice.transpose(1, 2)                    # (B, nx, nx) shared gram
    Ginv = torch.linalg.inv(G)
    s = X_slice.sum(-1)                                      # (B, nx)
    Rmat = RR @ X_slice.transpose(1, 2)                      # (B, nx, nx), row i = R_i . X
    RS1 = RR.sum(-1)                                         # (B, nx)
    RS2 = (RR**2).sum(-1)                                    # (B, nx)

    gs = (Ginv @ s.unsqueeze(-1)).squeeze(-1)               # (B, nx) = Ginv s  (shared)
    GR = Ginv @ Rmat.transpose(1, 2)                        # (B, nx, nx), col i = Ginv r_i
    sGs = (s * gs).sum(-1)                                   # (B,)
    sGr = (Rmat @ gs.unsqueeze(-1)).squeeze(-1)            # (B, nx)
    rGr = torch.einsum('bik,bki->bi', Rmat, GR)             # (B, nx)

    # 2x2 capacitance matrix  Cap = P - (1/alpha) W^T Ginv W  (per b, i)
    c00 = N * alpha - alpha * sGs.unsqueeze(-1)
    c01 = (2 * dt / bb**3) * RS1 - (2 * dt / bb**3) * sGr
    c11 = (3 * dt / bb**4 * RS2 - N / bb**2) - (4 * dt / bb**4) * rGr
    det = c00 * c11 - c01**2
    Ci00, Ci01, Ci11 = c11 / det, -c01 / det, c00 / det     # inverse of the 2x2

    # diag(S_i^{-1}) = (1/alpha_i) diag(Ginv) + diag( Z_i Cap_i^{-1} Z_i^T )
    # with Z_i = [ Ginv s ,  (2/b_i) Ginv r_i ]
    z1 = (2.0 / bb).unsqueeze(-1) * GR.transpose(1, 2)      # (B, i, k)
    z0 = gs.unsqueeze(1)                                     # (B, 1, k) shared over i
    diag_corr = (Ci00.unsqueeze(-1) * z0**2
                 + 2 * Ci01.unsqueeze(-1) * z0 * z1
                 + Ci11.unsqueeze(-1) * z1**2)
    term1 = (1.0 / alpha).unsqueeze(-1) * torch.diagonal(Ginv, dim1=1, dim2=2).unsqueeze(1)
    diag_ps = term1 + diag_corr                              # (B, i, k)

    cdiag = torch.diagonal(C, dim1=1, dim2=2)                # (B, nx)
    return (C / cdiag.unsqueeze(-1))**2 * diag_ps           # (B, i, j)


def simulate_ode_vectorized(n_steps, initial_state, dt, sigma, device='cpu'):
    """
    Vectorized ODE simulation for GPU acceleration.
    Simulates: dxdt = y + noise, dydt = -y + noise
    
    Args:
        n_steps: number of time steps
        initial_state: [x0, y0]
        dt: time step
        sigma: noise level
        device: 'cpu' or 'cuda'
    
    Returns:
        xy: tensor of shape (n_steps+1, 2)
    """
    xy = torch.zeros((n_steps + 1, 2), dtype=torch.float32, device=device)
    xy[0] = torch.tensor(initial_state, dtype=torch.float32, device=device)
    
    # Generate all random noise at once for better GPU utilization
    noise = sigma * torch.randn((n_steps, 2), dtype=torch.float32, device=device)
    
    for i in range(n_steps):
        xy[i+1, 0] = xy[i, 0] + xy[i, 1] * dt + noise[i, 0]
        xy[i+1, 1] = xy[i, 1] - xy[i, 1] * dt + noise[i, 1]
    
    return xy

def causal_est_matrix(X, n_step=1, dt=1, device=None):
    '''
    input: 
        X: a 2D torch tensor of shape (C,T), each ROW is a time series
        n_step: the number of steps to calculate the derivative (Euler forward) 
        dt: the time interval between two time points
        device: torch device to use (e.g., 'cuda' or 'cpu')
    returns:
        causal_matrix: a 2D torch tensor, (i,j) entry is the causality from i to j
        var: a 2D torch tensor, the variance of the causality matrix
        c_norm: a 2D torch tensor, the normalized causality matrix
    '''
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    
    if device is not None:
        X = X.to(device)
    
    nx, nt = X.shape
    
    # Get covariance matrix
    X_centered = X[:,:-n_step] - X[:,:-n_step].mean(dim=1, keepdim=True)
    C = (X_centered @ X_centered.T) / (nt - 1 - n_step)

    # Get derivative and its covariance
    dX = (X[:, n_step:] - X[:, :-n_step]) / (n_step * dt)
    dX_centered = dX - dX.mean(dim=1, keepdim=True)
    dC = (X_centered @ dX_centered.T) / (nt - 1 - n_step)

    # Calculate causality matrix
    try:
        T_pre = torch.linalg.solve(C, dC)
    except:
        T_pre = torch.linalg.pinv(C) @ dC

    C_diag = torch.diag(1.0/torch.diag(C))
    cM = (C @ C_diag) * T_pre

    # Calculate residuals
    ff = dX.mean(dim=1) - (X[:,:-n_step].mean(dim=1, keepdim=True).T @ T_pre).squeeze()
    RR = dX - ff.unsqueeze(1) - T_pre.T @ X[:,:-n_step]

    QQ = torch.sum(RR**2, dim=-1)
    bb = torch.sqrt(QQ*dt/(nt-n_step))
    dH_noise = bb**2/2/torch.diag(C)

    # Normalization
    ZZ = torch.sum(torch.abs(cM), dim=0, keepdim=True) + torch.abs(dH_noise.unsqueeze(0))
    cM_Z = cM / ZZ

    # Variance estimation via the Fisher information matrix (Schur/Woodbury form,
    # see _variance_diag): avoids building/inverting the (nx, nx+2, nx+2) tensor.
    N = nt - 1
    var = _variance_diag(C.unsqueeze(0), X[:, :-n_step].unsqueeze(0),
                         RR.unsqueeze(0), bb.unsqueeze(0), dt, N)[0]

    return cM, var.T, cM_Z

class LiangCausalityEstimator(torch.nn.Module):
    def __init__(self, n_step=1, dt=1):
        super().__init__()
        self.n_step = n_step
        self.dt = dt

    def forward(self, x):
        # x should be shape (batch, channels, time)
        # Optimized: process all batches using vectorized operations
        batch_size, nx, nt = x.shape
        device = x.device
        n_step = self.n_step
        dt = self.dt
        
        # Vectorized computation across batch dimension
        X_slice = x[:, :, :-n_step]  # (batch, nx, nt-n_step)
        X_centered = X_slice - X_slice.mean(dim=2, keepdim=True)
        
        # Covariance matrix: (batch, nx, nx)
        C = torch.bmm(X_centered, X_centered.transpose(1, 2)) / (nt - 1 - n_step)
        
        # Derivative
        dX = (x[:, :, n_step:] - x[:, :, :-n_step]) / (n_step * dt)
        dX_centered = dX - dX.mean(dim=2, keepdim=True)
        dC = torch.bmm(X_centered, dX_centered.transpose(1, 2)) / (nt - 1 - n_step)
        
        # Solve for T_pre using batch operations
        try:
            T_pre = torch.linalg.solve(C, dC)
        except:
            T_pre = torch.bmm(torch.linalg.pinv(C), dC)
        
        C_diag_vals = 1.0 / torch.diagonal(C, dim1=1, dim2=2)
        C_diag = torch.diag_embed(C_diag_vals)
        cM = torch.bmm(C, C_diag) * T_pre
        
        # Calculate residuals
        ff = dX.mean(dim=2) - torch.bmm(X_slice.mean(dim=2).unsqueeze(1), T_pre).squeeze(1)
        RR = dX - ff.unsqueeze(2) - torch.bmm(T_pre.transpose(1, 2), X_slice)
        
        QQ = torch.sum(RR**2, dim=-1)
        bb = torch.sqrt(QQ * dt / (nt - n_step))
        dH_noise = bb**2 / 2 / torch.diagonal(C, dim1=1, dim2=2)
        
        # Normalization
        ZZ = torch.sum(torch.abs(cM), dim=1, keepdim=True) + torch.abs(dH_noise.unsqueeze(1))
        cM_Z = cM / ZZ
        
        # Variance estimation via the Fisher information matrix (Schur/Woodbury
        # form, see _variance_diag): one nx-by-nx inverse per batch item plus
        # closed-form 2x2 inverses, instead of allocating/inverting a
        # (batch, nx, nx+2, nx+2) tensor.
        N = nt - 1
        var = _variance_diag(C, X_slice, RR, bb, dt, N)

        return cM, var.transpose(1, 2), cM_Z

#%%
if __name__ == '__main__':
    import time
    from causality_estimation import causality_est_with_sig_norm

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    time_costs = []
    '''
    The ODE system 
    dxdt = y + noise 
    dydt = -y + noise
    '''
    t_span = (0, 100)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    dt = 0.1
    sigma = 0.01
    
    # Use vectorized GPU-accelerated simulation
    xy = simulate_ode_vectorized(
        n_steps=t_eval.shape[0],
        initial_state=[0, 1],
        dt=dt,
        sigma=sigma,
        device=device
    ).cpu().numpy()  # Convert to numpy for compatibility
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(t_eval, xy[1:,0], label='x')
    plt.plot(t_eval, xy[1:,1], label='y')
    plt.xlabel('t')
    plt.legend()
    
    #%%
    start_time = time.time()
    cau1, var1, cau1_normalized = causality_est_with_sig_norm(xy, n_step=1, dt=0.1)
    time_costs.append(time.time() - start_time)
    print('Causality matrix:')
    print(cau1.T)
    print('Variance matrix:')
    print(var1.T)
    print('Normalized causality matrix:')
    print(cau1_normalized.T)
    print('Significant test:')
    print((np.abs(cau1)>np.sqrt(var1)*2.56).T)
    print('Time cost:', time_costs[-1])

    xy_torch = torch.tensor(xy, dtype=torch.float32, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()  # Ensure all ops are complete
    start_time = time.time()
    cau2, var2, cau2_normalized = causal_est_matrix(xy_torch.T, n_step=1, dt=0.1, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()  # Wait for GPU to finish
    time_costs.append(time.time() - start_time)
    print('Causality matrix:')
    print(cau2.cpu() if device == 'cuda' else cau2)  # the result is the transpose of the above result
    print('Variance matrix:')
    print(var2) # the result is the transpose of the above result
    print('Normalized causality matrix:')
    print(cau2_normalized) # the result is the transpose of the above result
    print('Significant test:')
    print((torch.abs(cau2)>torch.sqrt(var2)*2.56).cpu())
    print('Time cost:', time_costs[-1])
    #%%
    '''
    toy example in the reference
    '''
    alpha = np.array([0.1,0.7,0.5,0.2,0.8,0.3]).T
    A = np.array([[0,0,-0.6,0,0,0],
                  [-0.5,0,0,0,0,0.8],
                  [0,0.7,0,0,0,0],
                  [0,0,0,0.7,0.4,0],
                  [0,0,0,0.2,0,0.7],
                  [0,0,0,0,0,-0.5]]) # Aij indicates the influence from j to i
    mu=0;sigma=1
    B = np.zeros_like(A)
    for i in range(6):
        B[i,i]=1

    x = np.empty((10000,6))
    #initialization
    x[0] = np.random.normal(mu,sigma,6)

    for i in range(1,10000):
        x[i] = alpha+A@x[i-1]+np.random.multivariate_normal(np.array([0,0,0,0,0,0]),B)
    
    start_time = time.time()
    causality,variance,normalized_causality = causality_est_with_sig_norm(x)
    time_costs.append(time.time() - start_time)
    print(np.round(causality.T,2))
    print((np.abs(causality)>np.sqrt(variance)*2.56).T)
    # print(np.round(normalized_causality.T,3))
    print('Time cost:', time_costs[-1])
    
    print('-------------------\n')
    xx = torch.tensor(x.T, dtype=torch.float32, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    causality1,variance1,normalized_causality1 = causal_est_matrix(xx, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_costs.append(time.time() - start_time)
    print(np.round(causality1.cpu() if device == 'cuda' else causality1,2))
    print((torch.abs(causality1)>torch.sqrt(variance1)*2.56).cpu())
    # print(np.round(normalized_causality,3))
    print('Time cost:', time_costs[-1])

    #%% 
    '''
    additional example
     # aij indicates the influence from j to i
    a11=0;   a21=0;   a31=-0.6;  a41=0;   a51=-0.0;  a61=0;   b1=0.1;
    a12=-0.5;a22=0;   a32=-0.0;  a42=0;   a52=0.0;   a62=0.8; b2=0.7;
    a13=0;   a23=0.7; a33=-0.6;  a43=0;   a53=-0.0;  a63=0;   b3=0.5;
    a14=0;   a24=0;   a34=-0.;   a44=0.7; a54=0.4;   a64=0;   b4=0.2;
    a15=0;   a25=0;   a35=0;     a45=0.2; a55=0.0;   a65=0.7; b5=0.8;
    a16=0;   a26=0;   a36=0;     a46=0;   a56=0.0;   a66=-0.5;b6=0.3;
    '''
    xx=np.loadtxt('./example_data/case2_data.txt') # (100001, 6)

    xx=xx[10000:].T # (100001, 6) -> (6, 90001)

    start_time = time.time()
    cau, var, cau_normalized = causality_est_with_sig_norm(xx.T, n_step=1, dt=1)
    time_costs.append(time.time() - start_time)
    print('Causality matrix:')
    print(cau)
    print('Variance matrix:')
    print(var)
    print('Normalized causality matrix:')
    print(cau_normalized)
    print('Significant test:')
    print((np.abs(cau)>np.sqrt(var)*2.56))
    print('Time cost:', time_costs[-1])
    
    print('-------------------\n')
    xx = torch.tensor(xx, dtype=torch.float32, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    cau4, var4, cau4_normalized = causal_est_matrix(xx, n_step=1, dt=1, device=device)
    if device == 'cuda':
        torch.cuda.synchronize()
    time_costs.append(time.time() - start_time)
    print('Causality matrix:')
    print((cau4.T.cpu() if device == 'cuda' else cau4.T))
    print('Variance matrix:')
    print(var4.T)
    print('Normalized causality matrix:')
    print(cau4_normalized.T)
    print('Significant test:')
    print((torch.abs(cau4)>torch.sqrt(var4)*2.56).T.cpu())
    print('Time cost:', time_costs[-1])

    # %%
    '''
    Usage as a torch module
    '''
    model = LiangCausalityEstimator(n_step=1, dt=1).to(device)
    xx = xx.unsqueeze(0)  # Add batch dimension
    causality, variance, normalized = model(xx)
    print('Model output causality matrix:')
    print(causality[0].T)
    print('Model output variance matrix:')
    print(variance[0].T) 
    print('Model output normalized matrix:')
    print(normalized[0].T)
    print('Model output significant test:')
    print((torch.abs(causality)>torch.sqrt(variance)*2.56)[0].T)
    
# %%
