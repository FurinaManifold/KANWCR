''''
Generate Data of    dX_t = drift(X_t) dt + diffusion(X_t) dB_t,  0<=t<=1
time_instants: E.g. torch.tensor([0, 0.2, 0.5, 1])
samples_num: E.g. 10000
dim: E.g. 1
drift_term: E.g. torch.tensor([0, 1, 0, -1]) -- that means drift = x - x^3
diffusion_term: E.g. torch.tensor([1, 0, 0, 0]) -- that means diffusion = 1
return data: [time, samples, dim]
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
#import utils


class DataSet(object):
    def __init__(self, time_instants, dt, samples_num, dim, drift_fun, diffusion_fun,
                 initialization):
        '''
        time_instants : Query time point
        dt: dt of generating process
        sample_num: calculated trajectory num
        dim: dimension of SDE
        drift fun: a function of input size: sample, dim  output: sample dim
        diffusion fun : a function of input size : sample dim output sample ,dim ,dim
        initialization: initial distribution of problem: shape sample , dim

        '''
        self.time_instants = time_instants
        self.dt = dt
        self.samples_num = samples_num
        self.dim = dim
        self.drift_fun = drift_fun
        self.diffusion_fun = diffusion_fun
        self.initialization = initialization
        
    def subSDE(self, t0, t1, x):
        '''
        drift fun: input: sample dim; output: sample dim
        diffusion fun: input sample dim; output:sample dim dim
        '''
        if t0 == t1:
            return x
        else:
            t = torch.arange(t0, t1 + self.dt, self.dt)
            y = x
            for i in range(t.shape[0] - 1):
                y = y + self.drift_fun(y) * self.dt + torch.einsum('bij,bjk->bi',self.diffusion_fun(y),torch.randn(x.shape[0],x.shape[1],1).cuda())* torch.sqrt(torch.tensor(self.dt).cuda())
                
            return y

    #@utils.timing
    def get_data(self, plot_hist=True):
        data = torch.zeros(self.time_instants.shape[0], self.samples_num, self.dim).cuda()
        data[0, :, :] = self.subSDE(0, self.time_instants[0], self.initialization)  # self.initialization
        for i in range(self.time_instants.shape[0] - 1):
            data[i + 1, :, :] = self.subSDE(self.time_instants[i], self.time_instants[i + 1], data[i, :, :])
        if plot_hist:
            for i in range(self.dim):
                plt.figure()
                plt.hist(x=data[-1, :, i].detach().cpu().numpy(), bins=80, range=[data.min().detach().cpu().numpy(), data.max().detach().cpu().numpy()], density=True)
                plt.xlim(-2.5,2.5)
                plt.ylim(0,0.6)
                wandb.log({f"data_distribution_dim{i}" : wandb.Image(plt)})
        return data




if __name__ == '__main__':
    def drift(x):
        return x - x ** 3
    def diffusion(x):
        batch = x.shape[0]
        dim = x.shape[1]
        diag = torch.eye(dim,dim).unsqueeze(0).repeat(batch,1,1).cuda()
        return diag
    dim = 2
    sample = 10000
    dataset = DataSet(torch.linspace(0,1,10).cuda(), dt=0.001, samples_num=sample, dim=dim,
                      drift_fun=drift, diffusion_fun=diffusion, initialization=torch.normal(mean=0., std=0.3, size=(sample, dim)).cuda())
    data = dataset.get_data(plot_hist=True)
    print("data.size: ", data.size())
    print("data.max: ", data.max(), "data.min: ", data.min())