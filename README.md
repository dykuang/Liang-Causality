# Liang-Causality
 A repo for implementing causality matrix with the method of Liang's information flow.

* `causality_estimation.py` contains the implementation using `numpy`.    
* `Causality_est_torch.py` contains the implementation using `torch`, where the method can be accessed via a regular function interface or as a torch module that can be directly integrated in neural networks' design as a layer.    

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
* variance matrix for the estimation `var` whose $ij$'s entry is the variance for estimated causality value of $ij$'s entry in `cau` and normalized cuasality matrix `cau_normalized`.

Here `causal_est_matrix` is from `causality_estimation.py`, $n_step$ gives the steps user specified for Euler finite difference scheme for estimation of derivatives in the internal algorithm. A further statistical test as below:
```
np.abs(cau)>np.sqrt(var)*2.56
```
provides the causality matrix (about 98.96% $\approx$ 99% significant level for about 2.56 standard deviation away from 0) as a boolean type:
```
False, False  
True, True
```
It actually reveals the true dynamics under the observation since the data is generated by the stochastic system ($\epsilon_1$ and $\epsilon_1$ are 1d Brownian motions):  
$\frac{dx}{dt} = y + \epsilon_1; \frac{dy}{dt} = -y + \epsilon_2$. Why? Because the first equation indicates the dynamic of $x$ is internally driven by $y$ alone $y\rightarrow x$ and the second equation tells that $y$ is driven by itself $y\rightarrow y$. Above causality is naturally: 

False ($x\rightarrow x$), False ($x\rightarrow y$)  
True ($y\rightarrow y$), True ($y\rightarrow x$)  

As a simple summary for linear ODE as $\frac{dX}{dt} = AX + \epsilon$, if $A_{ij} \neq 0$, then you should expect the boolean type causality matrix $C^B$ with $C_{ji}^B$ equals `True`. Further, the float type causality matrix $C^F$'s $ji$'s th entry $C_{ji}^F$ should have the same sign as $A_{ij}$. You can check this with the above example where $A$ is 
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
They should provide the same result. Please find some more examples in `examples.ipynb`.


## Reference
Please kindly cite this repository and following papers if you find this repo helpful in your project.  
    📑 X.San Liang, "[Normalized Multivariate Time Series Causality Analysis and Causal Graph Reconstruction](https://www.mdpi.com/1099-4300/23/6/679)", (2021)   
    📑 X.San Liang, "[Information flow and causality as rigorous notions ab initio](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.052201)",(2016)  
    📑 X.San Liang, "[Normalizing the causality between time series](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.92.022126)",(2015)  
    📑 X.San Liang, "[Unraveling the cause-effect relation between time series](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.90.052150)",(2014)  
    📑 X.San Liang, "[The Liang-Kleeman Information Flow: Theory and Applications](https://www.mdpi.com/1099-4300/15/1/327)",(2013)  
    📑 X.San Liang, Richard Kleeman, "[Information Transfer between Dynamical System Components](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.95.244101)",(2005)
    
