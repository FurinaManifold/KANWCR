import torch
import torch.nn as nn
import numpy as np
# import utils



class Gaussian(torch.nn.Module): 
    def __init__(self, mu, sigma, device):
        super(Gaussian, self).__init__()
        '''
        mu: gauss, time, dim
        sigma: gauss, dim
        '''
        self.mu = mu.to(device).unsqueeze(2)  # [gauss_num, time, 1, dim]
        self.sigma = sigma.unsqueeze(1).unsqueeze(1) # [gauss_num,1,1, dim]
        self.gauss_num, self.dim = mu.shape[0],mu.shape[2]
        self.device = device
        self.g0 = None

    def gaussZero(self, x):
        #print(x.shape,self.mu.shape,self.sigma.shape)
        """x: [t, sample, dim], return [gauss , t, sample]"""
        func = 1
        for d in range(self.dim):
            func = func * 1 / (self.sigma * torch.sqrt(2 * torch.tensor(torch.pi))) * torch.exp(
                -0.5 * (x[:, :, d].unsqueeze(0) - self.mu[:, :, :, d]) ** 2 / self.sigma ** 2)
        return func

    def gaussFirst(self, x, g0):
        """x: [t, sample, dim], g0: [gauss, t, sample], return[gauss, t, sample, dim]"""
        func = torch.zeros(self.gauss_num, x.shape[0], x.shape[1], x.shape[2])
        for d in range(self.dim):
            func[:, :, :, d] = -(x[:, :, d].unsqueeze(0) - self.mu[:, :, :, d])/self.sigma**2 * g0
        return func

    def gaussSecond(self, x, g0):
        """x: [t, sample, dim], g0: [gauss, t, sample, 1], return[gauss, t, sample, dim, dim]"""
        func = torch.zeros(self.gauss_num, x.shape[0], x.shape[1], x.shape[2])
        for k in range(x.shape[2]):
            func[:, :, :, k] = (
                            -1/self.sigma**2 + (-(x[:, :, k].unsqueeze(0)-self.mu[:, :, :, k])/self.sigma**2)
                            * (-(x[:, :, k].unsqueeze(0)-self.mu[:, :, :, k])/self.sigma**2)
                            ) * g0
        return func
    
    def forward(self, x, diff_order=0):
        if self.g0 is None:
            self.g0 = self.gaussZero(x).to(self.device)
        if diff_order == 0:
            return self.g0
        elif diff_order == 1:
            return self.gaussFirst(x, self.g0).to(self.device)
        elif diff_order == 2:
            return self.gaussSecond(x, self.g0).to(self.device)
        else:
            raise RuntimeError("higher order derivatives of the gaussian has not bee implemented!")
        
class Bump(torch.nn.Module):
    def __init__(self, mu, r, device): #f(x)=1_{|x-mu|<r}exp(-1/(1-|x-mu|^2/r^2))
        super().__init__()
        '''
        mu: gauss, time, dim
        r: gauss, dim
        '''
        self.mu = mu.unsqueeze(2).to(device)  # [bump_num, time,  dim]
        self.r = r.to(device) # [bump_num]
        self.bump_num, self.dim = mu.shape[0],mu.shape[-1]
        self.device = device
        self.f0 = None

    def stdbump(self, t):
        return torch.where(t>=1,0,torch.exp(-1/(1-t ** 2)))

    def bumpZero(self, x):
        #print(x.shape,self.mu.shape,self.sigma.shape)
        """x: [t, sample, dim], return [bump , t, sample]"""
        
        #func = torch.zeros([self.bump_num,x.shape[0],x.shape[1]])
        #print(func.shape)
        #print("bump_num={}".format(self.bump_num),"dim={}".format(self.dim))
        #print(self.stdbump(torch.norm(x.unsqueeze(0)-self.mu.unsqueeze(2),dim=3)/self.r.unsqueeze(1).unsqueeze(1)))
        #tmp=x.unsqueeze(0)-self.mu.unsqueeze(2)
        #print(x.unsqueeze(0).shape,self.mu.unsqueeze(2).shape,tmp.shape)
        #print (torch.norm(x.unsqueeze(0)-self.mu.unsqueeze(2),dim=3))
        return self.stdbump(torch.norm(x.unsqueeze(0)-self.mu,dim=3)/self.r.unsqueeze(1).unsqueeze(1))
        #for bump in range(self.bump_num):
        #    for t in range(x.shape[0]):
        #        for sample in range(x.shape[1]):
        #            func[bump,t,sample]=self.stdbump(torch.norm(x[t,sample,:]-self.mu[bump,t,:])/self.r[bump]) 

    def bumpFirst(self, x, f0): #\p_i f(x)=f(x)*(-2r^2(x_i-mu_i)/(r^2-|x-mu|^2)^2)
        """x: [t, sample, dim], f0: [bump, t, sample], return[bump, t, sample, dim]"""
        func = torch.zeros(self.bump_num, x.shape[0], x.shape[1], x.shape[2]).to(self.device)
        func[:,:,:,:]=-2*self.r.unsqueeze(1).unsqueeze(1).unsqueeze(1)**2*(x.unsqueeze(0)-self.mu)/(self.r.unsqueeze(1).unsqueeze(1).unsqueeze(1)**2-torch.norm(x.unsqueeze(0)-self.mu,dim=3).unsqueeze(3)**2)**2*f0.unsqueeze(3)
        #for bump in range(self.bump_num):
        #    for t in range(x.shape[0]):
        #        for sample in range(x.shape[1]):
        #            for d in range(self.dim):
        #                func[bump, t, sample, d] = -2*self.r[bump]**2*(x[t, sample, d]- self.mu[bump, t, d])/(self.r[bump]**2-(torch.norm(x[t,sample,:]-self.mu[bump,t,:]))**2)**2 * f0[bump,t,sample]
        return func

    def bumpSecond(self, x, f0):#\p_{ij} f(x)=f(x)*(-2\delta_{ij} r^2/(r^2-|x-mu|^2)^2-8r^2(x_i-mu_i)(x_j-mu_j)/(r^2-|x-mu|^2)^3+4r^4(x_i-mu_i)(x_j-mu_j)/(r^2-|x-mu|^2)^4)
        """x: [t, sample, dim], f0: [bump, t, sample], return[bump, t, sample, dim, dim]"""
        #only compute diagonal terms, so return [bump, t, sample, dim]
        func = torch.zeros(self.bump_num, x.shape[0], x.shape[1], self.dim).to(self.device)
        rr = self.r.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        dxmu = rr - torch.norm(x.unsqueeze(0)-self.mu,dim=3).unsqueeze(3)**2
        func[:,:,:,:] = 4*rr**2*(x.unsqueeze(0) - self.mu)**2 * f0.unsqueeze(3)*(-2/dxmu**3+rr**2/dxmu**4)-2*rr**2/dxmu**2 * f0.unsqueeze(3)
        return func
    
    def forward(self, x, diff_order=0):
        if self.f0 is None:
            self.f0 = self.bumpZero(x).to(self.device)
        if diff_order == 0:
            return self.f0
        elif diff_order == 1:
            return self.bumpFirst(x, self.f0).to(self.device)
        elif diff_order == 2:
            return self.bumpSecond(x, self.f0).to(self.device)
        else:
            raise RuntimeError("higher order derivatives of the bump function has not been implemented!")


class Gaussian_TD(torch.nn.Module): 
    def __init__(self, mu, sigma, device):
        """mu: [gauss_num,t, dim]; sigma: [gauss_num]"""
        super(Gaussian_TD, self).__init__()
        self.mu = mu.to(device).unsqueeze(2)   # [gauss_num, t, 1, dim]
        self.sigma = sigma.unsqueeze(1).unsqueeze(1) #gauss_num,1,1
        self.gauss_num,self.t_num, self.dim = mu.shape
        self.device = device
        self.g0 = None

    def gaussZero(self, x):
        """x: [t, sample, dim],
         self.mu: [gauss,t,1,dim] 
         return [gauss, t, sample]"""
        func = 1
        for d in range(self.dim):
            func = func * 1 / (self.sigma * torch.sqrt(2 * torch.tensor(torch.pi))) * torch.exp(
                -0.5 * (x[:, :, d].unsqueeze(0) - self.mu[:, :, :, d]) ** 2 / self.sigma ** 2)
        return func

    def gaussFirst(self, x, g0):
        """x: [t, sample, dim], 
        g0: [gauss, t, sample], 
        return[gauss, t, sample, dim]"""

        func = torch.zeros(self.gauss_num, x.shape[0], x.shape[1], x.shape[2])
        for d in range(self.dim):
            func[:, :, :, d] = -(x[:, :, d].unsqueeze(0) - self.mu[:, :, :, d])/self.sigma**2 * g0
        return func

    def gaussSecond(self, x, g0):
        """x: [t, sample, dim], 
        g0: [gauss, t, sample], 
        
        return[gauss, t, sample, dim, dim]"""
        func = torch.zeros(self.gauss_num, x.shape[0], x.shape[1], x.shape[2], x.shape[2])
        for k in range(x.shape[2]):
            for j in range(x.shape[2]):
                if k == j:
                    func[:, :, :, k, j] = (
                                    -1/self.sigma**2 + (-(x[:, :, k].unsqueeze(0)-self.mu[:, :, :, k])/self.sigma**2)
                                    * (-(x[:, :, j].unsqueeze(0)-self.mu[:, :, :, j])/self.sigma**2)
                                    ) * g0
                else:
                    pass
        return func
    
    def forward(self, x, diff_order=0):
        if self.g0 is None:
            self.g0 = self.gaussZero(x).to(self.device)
        if diff_order == 0:
            return self.g0
        elif diff_order == 1:
            return self.gaussFirst(x, self.g0).to(self.device)
        elif diff_order == 2:
            return self.gaussSecond(x, self.g0).to(self.device)
        else:
            raise RuntimeError("higher order derivatives of the gaussian has not been implemented!")

if __name__=="__main__":
    #print(torch.norm(torch.tensor([[1.,1.]])))
    b=Bump(torch.zeros([1,2,2]).double(),torch.ones([1]).double(),"cpu")
    print(b(torch.tensor([[[0.5,0.6]],[[0.7,0.8]]]).double(),diff_order = 0))
    print(b(torch.tensor([[[0.5,0.6]],[[0.7,0.8]]]).double(),diff_order = 1))
    print(b(torch.tensor([[[0.5,0.6]],[[0.7,0.8]]]).double(),diff_order = 2))
    




    
    