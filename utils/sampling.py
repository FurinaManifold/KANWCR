import torch
import torch.nn as nn
import numpy as np
from torch import optim
from utils.utils import timing
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import pyDOE

def rbf_kernel(X, h=-1):
    XY = np.dot(X, X.T)
    X2_ = np.sum(X**2, axis=1).reshape(-1, 1)
    X2 = X2_ + X2_.T - 2*XY
    if h < 0:  # Median heuristic for bandwidth
        h = np.median(X2) / (2 * np.log(X.shape[0] + 1))
    K = np.exp(-X2 / h)
    return K

# Gradient of the RBF Kernel
def grad_rbf_kernel(X, K, h=-1):
    XY = np.dot(X, X.T)
    X2_ = np.sum(X**2, axis=1).reshape(-1, 1)
    X2 = X2_ + X2_.T - 2*XY
    if h < 0:  # Median heuristic for bandwidth
        h = np.median(X2) / (2 * np.log(X.shape[0] + 1))
    dim = X.shape[1]
    dK = np.zeros((X.shape[0], X.shape[0], dim))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            dK[i, j, :] = K[i, j] * (X[i, :] - X[j, :]) / h
    return dK

# SVGD Update
def svgd_update(particles, grad_log_p, stepsize=0.1, num_iter=100):
    for _ in range(num_iter):
        # Compute kernel and its gradient
        K = rbf_kernel(particles)
        dK = grad_rbf_kernel(particles, K)

        # Compute SVGD gradient
        grad_logp = grad_log_p(particles)
        phi = np.zeros_like(particles)
        for i in range(particles.shape[0]):
            phi[i, :] = np.sum(K[i, :, np.newaxis] * grad_logp + dK[i, :, :], axis=0)

        # Update particles
        particles += stepsize * phi / particles.shape[0]
    return particles

# Example log probability and gradient (e.g., standard normal distribution)
def log_p(X):
    return -0.5 * np.sum(X**2, axis=1)

def grad_log_p(X):
    return -X





@timing
def sampleTestFunc_all(data, samp_number, mean_samp_way, var_samp_way, samp_coef,device):
    '''

    '''
    t_number = data.shape[0]
    n = data.shape[1] # 10000
    if mean_samp_way == 'lhs':
        
        length = samp_coef['lhs_ratio'] * (data.max() - data.min())
        mu_list = torch.rand(samp_number, device=device) * length - length / 2 + (data.max() + data.min()) / 2
        mu_list_all = mu_list.unsqueeze(1).repeat(1,t_number-2,1)

    if mean_samp_way == 'SDE_dist':

        index = np.arange(n) 
        np.random.shuffle(index)
        n_t = t_number-2 
        mu_list = data[:n_t,index[:samp_number],:].permute(1,0,2) #samp_number,n_t,dim
        mu_list = mu_list + torch.randn_like(mu_list)*0.02
        mu_list_all = mu_list # samp, n_t-2, dim
        # sample mu list done 
            
    if mean_samp_way == 'SVGD':
        data = data.detach().cpu().numpy()
        n_t = t_number-2 
        mu_list = torch.zeros(samp_number,n_t,data.shape[2])
        for i in range(n_t):
            mu_list[:,i,:] = svgd_update(data[i,:,:], grad_log_p, stepsize=0.1, num_iter=500)[:samp_number,:]
    
    if var_samp_way == 'constant':
        variance = samp_coef['variance_max']*torch.ones(t_number-2)
    elif var_samp_way == 'dist':
        mu_dist = 1/torch.norm(mu_list_all[:,5,:],p=2,dim = -1)**2
        mu_dist_min = torch.min(mu_dist)
        mu_dist_max = torch.max(mu_dist)
        #mu_dist_norm = (mu_dist_max-mu_dist)/(mu_dist_max-mu_dist_min) # samp nt-2 dim
        mu_dist_norm = (mu_dist-mu_dist_min)/(mu_dist_max-mu_dist_min) # samp nt-2 dim
        variance = mu_dist_norm*(samp_coef['variance_max']-samp_coef['variance_min'])+samp_coef['variance_min'] 

    return mu_list_all, variance
