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

This version makes the implementation in pytorch with GPU optimizations:

GPU Optimizations:
    - Vectorized batch processing in LiangCausalityEstimator
    - Parallel matrix inversions (no loops)
    - Device parameter for explicit GPU placement
    - Batch matrix operations (bmm) for parallel computation
    - Pre-generated noise for vectorized simulation
    - Proper CUDA synchronization for accurate timing
    
Performance Tips:
    - Use mixed precision (torch.cuda.amp) for 2x speedup on modern GPUs
    - Process multiple time series in batches using LiangCausalityEstimator
    - Keep data on GPU to minimize CPU-GPU transfers
    - Use torch.compile() in PyTorch 2.0+ for additional speedup
'''
#%%
import numpy as np
import torch

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

    # Variance estimation
    N = nt-1
    NNI = torch.zeros((nx, nx+2, nx+2), dtype=X.dtype, device=X.device)

    center = X[:,:-n_step] @ X[:,:-n_step].T
    RS1 = torch.sum(RR, dim=-1)
    RS2 = torch.sum(RR**2, dim=-1)

    center = dt/bb.unsqueeze(1).unsqueeze(2)**2 * center.unsqueeze(0)
    top_center = (dt/bb.unsqueeze(1)**2) @ torch.sum(X[:,:-n_step], dim=-1, keepdim=True).T
    right_center = (2*dt/bb.unsqueeze(1)**3) * (RR @ X[:,:-n_step].T)

    top_left_corner = N*dt/bb**2
    top_right_corner = 2*dt/bb**3*RS1
    bottom_right_corner = 3*dt/bb**4*RS2 - N/bb**2

    NNI[:,1:-1,1:-1] = center
    NNI[:,0,1:-1] = top_center
    NNI[:,1:-1,0] = top_center
    NNI[:,1:-1,-1] = right_center
    NNI[:,-1,1:-1] = right_center
    NNI[:,0,0] = top_left_corner
    NNI[:,0,-1] = top_right_corner
    NNI[:,-1,0] = top_right_corner
    NNI[:,-1,-1] = bottom_right_corner

    # Calculate inverse for all slices in parallel (GPU optimized)
    NNI_inv = torch.linalg.inv(NNI)  # Batch inversion
    diag_per_slice = torch.diagonal(NNI_inv, dim1=1, dim2=2)[:, 1:-1]  # Extract diagonals
    
    var = (C_diag @ C)**2 * diag_per_slice

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
        ff = dX.mean(dim=2) - torch.bmm(X_slice.mean(dim=2, keepdim=True), T_pre).squeeze(1)
        RR = dX - ff.unsqueeze(2) - torch.bmm(T_pre.transpose(1, 2), X_slice)
        
        QQ = torch.sum(RR**2, dim=-1)
        bb = torch.sqrt(QQ * dt / (nt - n_step))
        dH_noise = bb**2 / 2 / torch.diagonal(C, dim1=1, dim2=2)
        
        # Normalization
        ZZ = torch.sum(torch.abs(cM), dim=1, keepdim=True) + torch.abs(dH_noise.unsqueeze(1))
        cM_Z = cM / ZZ
        
        # Variance estimation (simplified for batch, can be expanded if needed)
        N = nt - 1
        NNI = torch.zeros((batch_size, nx, nx+2, nx+2), dtype=x.dtype, device=device)
        
        center = X_slice @ X_slice.transpose(1, 2)
        RS1 = torch.sum(RR, dim=-1)
        RS2 = torch.sum(RR**2, dim=-1)
        
        center = dt / bb.unsqueeze(2).unsqueeze(3)**2 * center.unsqueeze(1)
        top_center = (dt / bb.unsqueeze(2)**2) * torch.sum(X_slice, dim=-1, keepdim=True).unsqueeze(2)
        right_center = (2*dt / bb.unsqueeze(2)**3) * torch.bmm(RR, X_slice.transpose(1, 2)).unsqueeze(1)
        
        top_left_corner = N * dt / bb**2
        top_right_corner = 2 * dt / bb**3 * RS1
        bottom_right_corner = 3 * dt / bb**4 * RS2 - N / bb**2
        
        # Efficient batch indexing
        NNI[:, :, 1:-1, 1:-1] = center
        NNI[:, :, 0:1, 1:-1] = top_center.transpose(1, 2)
        NNI[:, :, 1:-1, 0:1] = top_center
        NNI[:, :, 1:-1, -1:] = right_center.transpose(1, 2)
        NNI[:, :, -1:, 1:-1] = right_center
        NNI[:, :, 0, 0] = top_left_corner
        NNI[:, :, 0, -1] = top_right_corner
        NNI[:, :, -1, 0] = top_right_corner
        NNI[:, :, -1, -1] = bottom_right_corner
        
        # Batch inverse
        NNI_flat = NNI.reshape(batch_size * nx, nx+2, nx+2)
        NNI_inv_flat = torch.linalg.inv(NNI_flat)
        NNI_inv = NNI_inv_flat.reshape(batch_size, nx, nx+2, nx+2)
        diag_per_slice = torch.diagonal(NNI_inv, dim1=2, dim2=3)[:, :, 1:-1]
        
        var = torch.bmm(C_diag, C)**2 * diag_per_slice
        
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
    print((np.abs(cau2)>np.sqrt(var2)*2.56))
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
    print((np.abs(causality1)>np.sqrt(variance1)*2.56))
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
    print((np.abs(cau4)>np.sqrt(var4)*2.56).T)
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
