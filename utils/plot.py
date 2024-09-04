import torch
import torch.nn as nn
import numpy as np
from torch import optim
from collections import OrderedDict
import time
import psutil
import os
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import utils
import itertools
import logging
logger = logging.getLogger(__name__)
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

def plot_error(drift_fun,drift_net,diffusion_fun,diffusion_net,device,plot_type):
    if plot_type == '1d':
        plot_1d(drift_fun,drift_net,diffusion_fun,diffusion_net,device)
    elif plot_type == '2d':
        plot_2d(drift_fun,drift_net,diffusion_fun,diffusion_net,device)

def plot_1d(drift_fun,drift_net,diffusion_fun,diffusion_net,device):
    interval=[-1.5,1.5]
    n=100
    x = torch.linspace(interval[0], interval[1], n, device=device).unsqueeze(1)
    exact_drift = drift_fun(x) # n,1
    exact_diffusion = diffusion_fun(x)[...,0] # n,1
    with torch.no_grad():
        pred_drift = drift_net(x)# n,1
        pred_diffusion = torch.sqrt(diffusion_net(x)) # n,1
    plt.figure()
    plt.plot(x.cpu().detach().numpy(), exact_drift.cpu().detach().numpy(), label="exact")
    plt.plot(x.cpu().detach().numpy(), pred_drift.cpu().detach().numpy(), label="approximate")
    plt.xlabel("x")
    plt.ylabel("drift")
    plt.legend()
    plt.show()
    img = wandb.Image(plt)
    wandb.log({'Image_drift': img})
    plt.savefig("save_files/Image_drift.png")
    plt.close()
    
    plt.figure()
    plt.plot(x.cpu().detach().numpy(), exact_diffusion.cpu().detach().numpy(), label="exact")
    plt.plot(x.cpu().detach().numpy(), pred_diffusion.cpu().detach().numpy(), label="approximate")
    plt.xlabel("x")
    plt.ylabel("diffusion")
    plt.ylim(bottom=0, top=2)
    plt.legend()
    img = wandb.Image(plt)
    wandb.log({'Image_diffusion': img})
    plt.savefig("save_files/Image_diffusion.png")
    plt.close()


def plot_2d(drift_fun,drift_net,diffusion_fun,diffusion_net,device):
    interval=[-1,1]
    n=50
    x = torch.linspace(interval[0], interval[1], n, device=device)
    y = x
    X,Y = torch.meshgrid(x,y)
    data = torch.cat((X.reshape(n,n,1),Y.reshape(n,n,1)),dim  = -1)
    exact_drift = torch.zeros(n,n,2)
    exact_diffusion = torch.zeros(n,n,2,2)
    pred_drift = torch.zeros(n,n,2)
    pred_diffusion = torch.zeros(n,n,2)

    #print("Xsize=",X.size())
    #print("Ysize=",Y.size())
    #print("datasize=",data.size())
    #print(data[0,0,:].unsqueeze(0))
    #print(drift_net(data[0,0,:].unsqueeze(0)))
    for i in range(n):
        for j in range(n):
            exact_drift[i,j,:]= drift_fun(data[i,j,:].unsqueeze(0))[0]
            exact_diffusion[i,j,:,:]=diffusion_fun(data[i,j,:].unsqueeze(0))[0]
            pred_drift[i,j,:]= drift_net(data[i,j,:].unsqueeze(0))[0]
            pred_diffusion[i,j,:]=diffusion_net(data[i,j,:].unsqueeze(0))[0]
    #exact_drift = drift_fun(data)# n,2
    #self.net.load_state_dict(torch.load('1D_drift.pth'))
    #exact_diffusion = diffusion_fun(data) # n,2,2

   
    #pred_drift = drift_net(data)# n,2
    #pred_diffusion = torch.sqrt(diffusion_net(data))# n,2

    plt.figure(figsize = (16,16))
    plt.subplot(2,2,1)
    levels = np.linspace(-4,4,51)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), exact_drift.detach().cpu().numpy()[...,0],80, cmap='rainbow',levels = levels)
    plt.title("exact_drift_X")
    plt.colorbar()
    #plt.savefig("results/exact1.png")
    #plt.show()
    #plt.close()
    plt.subplot(2,2,2)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), exact_drift.detach().cpu().numpy()[...,1],80, cmap='rainbow',levels = levels)
    plt.title("exact_drift_Y")
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), pred_drift.detach().cpu().numpy()[...,0],80, cmap='rainbow',levels = levels)
    plt.colorbar()
    plt.title("pred_drift_X")
    plt.subplot(2,2,4)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), pred_drift.detach().cpu().numpy()[...,1],80, cmap='rainbow',levels = levels)
    plt.title("pred_drift_Y")
    plt.colorbar()
    
    levelsdif = np.linspace(-2,2,51)
    plt.subplot(4,2,5)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), exact_diffusion.detach().cpu().numpy()[...,0,0],80, cmap='rainbow',levels=levelsdif)
    plt.title("exact_diffusion_X")
    plt.colorbar()
    #plt.savefig("results/exact1.png")
    #plt.show()
    #plt.close()
    plt.subplot(4,2,6)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), exact_diffusion.detach().cpu().numpy()[...,1,1],80, cmap='rainbow',levels=levelsdif)
    plt.title("exact_diffusion_Y")
    plt.colorbar()
    plt.subplot(4,2,7)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), pred_diffusion.detach().cpu().numpy()[...,0],80, cmap='rainbow',levels=levelsdif)
    plt.title("pred_diffusion_X")
    plt.colorbar()
    plt.subplot(4,2,8)
    plt.contourf(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), pred_diffusion.detach().cpu().numpy()[...,1],80, cmap='rainbow',levels=levelsdif)
    plt.title("pred_diffusion_Y")
    plt.colorbar()

    img = wandb.Image(plt)
    wandb.log({'Image': img})
    plt.savefig("save_files/Image_2d.png")
    plt.close()