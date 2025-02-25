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

'''
#%%
import numpy as np
def bivar_causality(XX, n_step=1, dt=1):
    '''
    Calculate the causality from X2 to X1 under linear assumption
    and additive noise 

    T_1->2 = (C22 * C21 * C1d2 - C21^2 * C2d2)/(C22^2 * C11 - C22 * C21^2)

    input:
        XX: two time series stacked, shape is (2, T)
        nstep: the number of steps to calculate the derivative (Euler forward) 
        dt: the time interval between two time points
    output:
        The causality from the first row to the second row
    '''
    
    '''
    Get covariance matrix of X
    '''
    # C = np.cov(XX)
    C = np.cov(XX[:,:-n_step]) # ? using this gives the same result as the above function
    '''
    Get the first derivative estimation
    '''
    dXX = (XX[:, n_step:] - XX[:, :-n_step]) / (n_step * dt)

    '''
    Get C1,d2
    '''
    C1d2 = np.cov(XX[0, :-n_step], dXX[1, :])[0,1]

    '''
    Get C2,d2
    '''
    C2d2 = np.cov(XX[1, :-n_step], dXX[1, :])[0,1]

    '''
    compute causality
    '''
    T12 = (C[1, 1] * C[1, 0] * C1d2 - C[1, 0]**2 * C2d2) / (C[1, 1]**2 * C[0, 0] - C[1, 1] * C[1, 0]**2)
    
    return T12


def causality_est_with_sig_norm(xx,n_step=1,dt=1):
    '''
    Original code from Jiwang Ma

    input:
        xx: a 2D numpy array of shape (T, C), each column is a time series
        n_step: the number of steps to calculate the derivative (Euler forward) 
        dt: the time interval between two time points
    returns:
        tmp: causality matrix, (i,j) entry is the causality from j to i
        var_T: estimated variance of the causality matrix
        prop: the normalized causality matrix
    '''
    
    nt,nv = xx.shape

    dx = (xx[n_step:,:]-xx[:-1,:])/(n_step*dt)
    x = xx[:-1,:]
    del(xx)

    entropy_transfer = np.zeros((nv,nv))
    entropy_generation = np.zeros((nv,nv))

    dC = np.zeros((nv,nv))
    x_tmp = x
    C = np.cov(x_tmp.T)        
    for j in range(nv): #j is locations of dx
        for i in range(nv): #i is locations of x
            dC[i,j] = np.sum((x[:,i]-np.mean(x[:,i]))*(dx[:,j]-np.mean(dx[:,j])))/(nt-2)
    aln = np.dot(np.linalg.inv(C),dC).T
    for j in range(nv): #j is locations of dx
        entropy_generation[j,j] = aln[j,j]
        for i in range(nv): #i is locations of x
            if i != j:
                entropy_transfer[i,j] = C[i,j]/C[i,i]*aln[i,j]

    f = np.mean(dx,axis=0)

    R = np.zeros_like(dx)
    for i in range(nv):
        f[i]=f[i]-np.dot(aln[i],np.mean(x,axis=0))
        R[:,i] = dx[:,i]-f[i]
        for j in range(nv):
            R[:,i] = R[:,i]-aln[i,j]*x[:,j]
    Q = np.sum(R*R,axis=0)
    b = np.sqrt(Q*dt/(nt-n_step)) 
    dH_noise = np.zeros_like(b)
    for i in range(nv):
        dH_noise[i] = b[i]*b[i]/(2.*C[i,i])

    tmp = entropy_generation+entropy_transfer
    prop = np.zeros_like(tmp)

    ###################normalization########################
    for i in range(nv):
        Z = np.sum(np.abs(tmp[i]))+np.abs(dH_noise[i])
        for j in range(nv):
            prop[i,j]=tmp[i,j]/Z       
    ###################估计方差###############################
    N = nt-1
    var_T = np.zeros((nv,nv))
    var_coef = np.zeros(nv)
    NI = np.zeros((nv, nv+2, nv+2))
    for i in range(nv):
        NI[i, 0, 0] = N * dt / (b[i] * b[i])
        NI[i, nv+1, nv+1] = 3 * dt / (b[i]*b[i]*b[i]*b[i]) * \
        np.sum(R[:,i]*R[:,i]) - N / (b[i]*b[i])
        for j in range(nv):
            NI[i,0,j+1]=dt/(b[i]*b[i])*np.sum(x[:,j])
        NI[i,0,nv+1] = 2*dt/(b[i]*b[i]*b[i])*np.sum(R[:,i])
        for k in range(nv):
            for j in range(nv):
                NI[i,j+1,k+1]=dt/(b[i]*b[i])*np.sum(x[:,j]*x[:,k])
        for j in range(nv):
            NI[i,j+1,nv+1]=2*dt/(b[i]*b[i]*b[i])*np.sum(R[:,i]*x[:,j])
        for j in range(nv+2):
            for k in range(j):
                NI[i,j,k]=NI[i,k,j]
        invNI = np.linalg.inv(NI[i])
        for j in range(nv):
            var_coef[j] = invNI[j+1,j+1]
        var_T[i] = ((C[i]/C[i,i])*(C[i]/C[i,i]))*var_coef
            

    return tmp,var_T,prop

def causal_est_matrix(X,n_step=1,dt=1):
    '''
    A revised version for faster computation
    input: 
        X: a 2D numpy array of shape (C,T), each ROW is a time series
        n_step: the number of steps to calculate the derivative (Euler forward) 
        dt: the time interval between two time points
    returns:
        causal_matrix: 
            a 2D numpy array, the causal matrix
            (i,j) entry is the causality from i to j
        var: 
            a 2D numpy array, the variance of the causality matrix
        c_norm: 
            a 2D numpy array, the normalized causality matrix
    '''

    '''
    Get covariance matrix of X
    '''
    nx, nt = X.shape
    # C = np.cov(X)
    C = np.cov(X[:,:-n_step]) # ? using this gives the same result as the above function

    '''
    Get sample covariance matrix of X and its derivative
    '''
    dX = (X[:, n_step:]- X[:, :-n_step])/(n_step*dt)

    # dC = np.cov(X[:,:-n_step], dX)
    dC = (X[:,:-n_step] - np.mean(X[:,:-n_step], axis=1, keepdims=True)) @ (dX - np.mean(dX, axis=1,keepdims=True)).T / (nt - 1 - n_step)
    # dC = (X[:,:-1] - np.mean(X[:,:-1], axis=1, keepdims=True)) @ (dX - np.mean(dX, axis=1,keepdims=True)).T / (X.shape[1] - 1 - n_step)
    
    # dC = (X[:,n_step:] - np.mean(X[:,n_step:], axis=1, keepdims=True)) @ (dX - np.mean(dX, axis=1,keepdims=True)).T / (X.shape[1] - 1)
    # dC = (X[:,np.floor(n_step//2):] - np.mean(X[:,np.floor(n_step//2):], axis=1, keepdims=True)) @ (dX - np.mean(dX, axis=1,keepdims=True)).T / (X.shape[1] - 1)
    # dC = (X - np.mean(X, axis=1, keepdims=True)) @ (dX - np.mean(dX, axis=1,keepdims=True)).T / (X.shape[1] - 1)
    
    '''
    get the causality matrix
    '''
    try:       
        T_pre = np.linalg.solve(C, dC) # can be replaced with faster iteration scheme
    except:
        T_pre = np.linalg.pinv(C) @ dC
    
    
    C_diag = np.diag(1.0/np.diag(C))
    # C_diag = np.eyes(T_pre.shape[0])*C)
    cM = (C@C_diag)*T_pre

    ff = np.mean(dX,axis=1) - (np.mean(X[:,:-1],axis=1,keepdims=True).T @ T_pre).squeeze()

    RR = dX - ff[:,None] - T_pre.T@X[:,:-1]

    QQ = np.sum(RR**2, axis=-1)
    bb = np.sqrt(QQ*dt/(nt-n_step))
    dH_noise = bb**2/2/np.diag(C)

    ###################  Normalization  ######################## 
    
    ZZ = np.sum(np.abs(cM), axis=0, keepdims=True) \
        + np.abs(dH_noise[None,:])
    cM_Z = cM / ZZ

    ###################  Variance estimation w Fisher Matrix ###############################
    N = nt-1

    NNI = np.zeros((nx,nx+2,nx+2))

    center = X[:,:-1] @ X[:,:-1].T
    RS1 = np.sum(RR, axis=-1)
    RS2 = np.sum(RR**2, axis=-1)

    center = dt/bb[:,None,None]**2 * center[None,...]
    top_center = (dt/bb[:,None]**2) @ np.sum(X[:,:-1], axis=-1,keepdims=True).T
    right_center = (2*dt/bb[:,None]**3) * ( RR @ X[:,:-1].T )

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

    inv_per_slice = list(map(np.linalg.inv, 
                             [NNI[i] for i in range(nx)]))
    diag_per_slice = [np.diag(inv_per_slice[i])[1:-1] for i in range(nx)]
    
    var = (C_diag @ C)**2 * np.array(diag_per_slice)

    return cM, var.T, cM_Z

#%%
if __name__ == '__main__':
    import time
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
        tmp_y = -2*xy[i][1]*dt + xy[i][1]
        xy[i+1][0] = (xy[i][1]+tmp_y)/2*dt + xy[i][0] + sigma*np.random.randn()
        xy[i+1][1] = -2*(xy[i][1]+tmp_y)/2*dt + xy[i][1] + sigma*np.random.randn()
    
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
    start_time = time.time()
    causality1,variance1,normalized_causality1 = causal_est_matrix(x.T)
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
    xx=np.loadtxt('./case2_data.txt') # (100001, 6)

    xx=xx[10000:].T # (100001, 6) -> (6, 90001)
    from multi_causality_est import multi_causality_est
    
    start_time = time.time()
    cau3=multi_causality_est(X=xx)#X [ndim,ntime_steps]
    time_costs.append(time.time() - start_time)

    #information flow from column to raw
    IF = np.squeeze(cau3['IF']) 
    #normalized information flow
    nIF= np.squeeze(cau3.get('nIF'))
    #significant test: confidence levels of 90/95/99% err_e90/err_e95/err_e99;
    err= np.squeeze(cau3.get('err_e99')) 
    #significant test (p-value)
    p=np.squeeze(cau3.get('p'))

    print('Causality matrix:')
    print(nIF)
    print('Significant test:')
    print((np.abs(nIF)>err))
    print('Time cost:', time_costs[-1])
    
    # print('-------------------\n')
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
