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

This version makes the implementation in pytorch
'''
#%%
import numpy as np
import torch

def causal_est_matrix(X, n_step=1, dt=1):
    '''
    input: 
        X: a 2D torch tensor of shape (C,T), each ROW is a time series
        n_step: the number of steps to calculate the derivative (Euler forward) 
        dt: the time interval between two time points
    returns:
        causal_matrix: a 2D torch tensor, (i,j) entry is the causality from i to j
        var: a 2D torch tensor, the variance of the causality matrix
        c_norm: a 2D torch tensor, the normalized causality matrix
    '''
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    
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
    NNI = torch.zeros((nx, nx+2, nx+2), dtype=X.dtype)

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

    # Calculate inverse for each slice
    diag_per_slice = []
    for i in range(nx):
        inv = torch.linalg.inv(NNI[i])
        diag_per_slice.append(torch.diag(inv)[1:-1])
    
    var = (C_diag @ C)**2 * torch.stack(diag_per_slice)

    return cM, var.T, cM_Z

class LiangCausalityEstimator(torch.nn.Module):
    def __init__(self, n_step=1, dt=1):
        super().__init__()
        self.n_step = n_step
        self.dt = dt

    def forward(self, x):
        # x should be shape (batch, channels, time)
        batch_size = x.shape[0]
        results = []
        for i in range(batch_size):
            cM, var, cM_Z = causal_est_matrix(x[i], self.n_step, self.dt)
            results.append((cM, var, cM_Z))
        
        # Stack results along batch dimension
        causality = torch.stack([r[0] for r in results])
        variance = torch.stack([r[1] for r in results])
        normalized = torch.stack([r[2] for r in results])
        
        return causality, variance, normalized

#%%
if __name__ == '__main__':
    import time
    from causality_estimation import causality_est_with_sig_norm

    device = 'cpu'

    time_costs = []
    '''
    The ODE system 
    dxdt = y + noise 
    dydt = -y + noise
    '''
    t_span = (0, 100)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    xy = np.zeros((t_eval.shape[0]+1,2))
    xy[0] = np.array([0,1]).T
    dt = 0.1
    sigma = 0.01

    for i in range(t_eval.shape[0]):
        xy[i+1][0] = xy[i][1] * dt + xy[i][0] + sigma*np.random.randn()
        xy[i+1][1] = -xy[i][1] * dt + xy[i][1] + sigma*np.random.randn()
    
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

    xy = torch.tensor(xy, dtype=torch.float32).to(device)
    start_time = time.time()
    cau2, var2, cau2_normalized = causal_est_matrix(xy.T, n_step=1, dt=0.1)
    time_costs.append(time.time() - start_time)
    print('Causality matrix:')
    print(cau2)  # the result is the transpose of the above result
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
    xx = torch.tensor(x.T, dtype=torch.float32)
    start_time = time.time()
    causality1,variance1,normalized_causality1 = causal_est_matrix(xx)
    time_costs.append(time.time() - start_time)
    print(np.round(causality1,2))
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
    xx = torch.tensor(xx, dtype=torch.float32)
    start_time = time.time()
    cau4, var4, cau4_normalized = causal_est_matrix(xx, n_step=1, dt=1)
    time_costs.append(time.time() - start_time)
    print('Causality matrix:')
    print(cau4.T)
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
    model = LiangCausalityEstimator(n_step=1, dt=1)
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
