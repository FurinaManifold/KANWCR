import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def model_select(model,frequency):
    if model == 'pdcd':
        drift = drift_poly3
        diffusion = diffusion_constant
    elif model == 'sdcd':
        drift = lambda x:2 * torch.sin(frequency * x)
        diffusion = diffusion_constant
    elif model == 'zx1':
        drift = lambda x:-(x-1)
        diffusion = diffusion_constant_sq2
    elif model == 'zx2':
        drift = drift_lor
        diffusion = diffusion_constant 
    elif model == 'zx3':
        drift = drift_real2d
        diffusion = lambda x:diffusion_diagonal(0.1,x)
    return drift, diffusion

def drift_lor(x):
    x0=x[...,0].unsqueeze(-1)
    x1=x[...,1].unsqueeze(-1)
    x2=x[...,2].unsqueeze(-1)
    return torch.cat([(x1-x0),x0*(1-x2)-x1,x0*x1-x2],dim = -1)

def drift_real2d(x):
    x0=x[...,0].unsqueeze(-1)
    x1=x[...,1].unsqueeze(-1)
    return torch.cat([x0*(1-x0**2)*0.2+x1*(1+torch.sin(x0)),-x1+2*x0*(1-x0**2)*(1+torch.sin(x0))],dim = -1)

def diffusion_diagonal(eps,x):
    diag = torch.tensor([[np.sqrt(0.4*eps),0],[0,np.sqrt(2*eps)]]).float().cuda()
    #print(diag.unsqueeze(0).repeat(x.shape[0],1,1).shape)
    return  diag.unsqueeze(0).repeat(x.shape[0],1,1)

def drift_sin(x):
    '''
    x:sample,dim
    out: sample,dim
    '''
    return 2 * torch.sin(8 * x)

def drift_poly3(x):
    '''
    x:sample,dim
    out: sample,dim
    '''
    return x - x ** 3

def diffusion_constant(x):
    '''
    n,1
    '''
    diag = torch.eye(x.shape[-1]).cuda()
    #print(diag.unsqueeze(0).repeat(x.shape[0],1,1).shape)
    return  diag.unsqueeze(0).repeat(x.shape[0],1,1)

def diffusion_constant_sq2(x):
    '''
    n,1
    '''
    diag = np.sqrt(2)*torch.eye(x.shape[-1]).cuda()
    #print(diag.unsqueeze(0).repeat(x.shape[0],1,1).shape)
    return  diag.unsqueeze(0).repeat(x.shape[0],1,1)

def diffusion_constant_sq40(x):
    '''
    n,1
    '''
    diag = np.sqrt(40)*torch.eye(x.shape[-1]).cuda()
    #print(diag.unsqueeze(0).repeat(x.shape[0],1,1).shape)
    return  diag.unsqueeze(0).repeat(x.shape[0],1,1)


def diffusion_knownnet(x):
    return  torch.ones_like(x)

def diffusion_knownnet_nonconst(x):
    return  torch.cat([np.sqrt(0.04)*torch.ones_like(x[...,0].unsqueeze(-1)),np.sqrt(0.2)*torch.ones_like(x[...,1].unsqueeze(-1))],dim=-1)

def diffusion_nonconstant(x):
    '''
    n,1
    '''
    diag = torch.eye(x.shape[-1]).cuda()
    val_const = torch.ones(x.shape[-2]).float().unsqueeze(-1).unsqueeze(-1).cuda()
    val = torch.ones(x.shape[-2]).float().unsqueeze(-1).unsqueeze(-1).cuda()+0.1 * torch.norm(x,dim = -1).unsqueeze(-1).unsqueeze(-1).cuda()**2
    #print(val.shape)
    return  diag.unsqueeze(0).repeat(x.shape[-2],1,1)*val_const

if __name__ == "__main__":
    print(drift_lor(torch.tensor([[[1,2,3],[4,5,6]]])))