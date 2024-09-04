import argparse 
import os
import warnings
import numpy as np
from utils.functions import model_select
from data.GenerateData_fun import DataSet
import torch
import yaml
from utils.utils import set_seed
from utils.getconfig import getcfg
import torch.utils.data as tud
from torch.utils.data import TensorDataset
from model.net import net_select
from model.test_fun import Gaussian

def main(config_file):
    torch.set_float32_matmul_precision("medium")
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    cfg_train = config['train']
    cfg_data = config['data']
    cfg_kan = config['KAN']
    print(cfg_kan)
    c_proj = config['Project']
    set_seed(config['seed'])
    # get definition of sde
    drift,diffusion = model_select(cfg_data['model'])

    # generate dataset   
    dt = cfg_data['dt'] # 生成数据的dt
    t = torch.linspace(0,cfg_data['T'],cfg_data['nt']).cuda()# 生成数据的时间轴
    dim = cfg_data['dim'] #问题维数
    samples = cfg_data['sample'] #生成轨道数量
    dataset = DataSet(t, dt=dt, samples_num=samples, dim=dim, drift_fun=drift, diffusion_fun=diffusion,
                      initialization=torch.normal(mean=0., std=cfg_train['sigma_init'], size=(samples, dim)))
    data = dataset.get_data(plot_hist=True) # t, sample_num ,dim
    Data = TensorDataset(data.unsqueeze(0))
    #print(data.shape)

    train_loader = tud.DataLoader(dataset = Data, batch_size = 1, shuffle = False)
    val_loader = tud.DataLoader(dataset = Data, batch_size = 1, shuffle = False)

    # get NN
    testFunc = Gaussian
    net_drift , net_diffusion = net_select(cfg_kan)
    # print(net_drift)
    x = torch.normal(0.0,1.0,(100,1))
    
    model = Model_pl(t = t, 
                     data = data, 
                     testFunc = testFunc, 
                     drift = drift, 
                     diffusion = diffusion, 
                     net_drift = net_drift,
                     net_diffusion = net_diffusion, 
                     cfg_train = cfg_train, 
                     cfg_kan = cfg_kan, 
                     device = data.device)
    





if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config_file = getcfg()
    main(config_file)