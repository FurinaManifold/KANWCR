from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import yaml
import argparse 
from bisect import bisect
import os
import torch
import shutil
import warnings
import wandb
wandb_logger = WandbLogger()
wandb.init(project='WCR_real2d_zx_KAN')
#wandb.init(mode = 'disabled')
import numpy as np
from data.GenerateData_fun import DataSet
from utils.utils import set_seed
from utils.functions import model_select
from model.test_fun import Gaussian,Bump
from model.net import net_select
from model.model import Model_pl
import torch.utils.data as tud
from utils.getconfig import getcfg
import torch
from torch.utils.data import TensorDataset

def main(config_file):
    torch.set_float32_matmul_precision("medium")
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    wandb_logger.experiment.config.update(config)
    cfg_train = config['train']
    cfg_data = config['data']
    cfg_nn = config['NN']
    print(f'cfg_nn:{cfg_nn}')
    c_proj = config['Project']
    set_seed(config['seed'])
    #device = config['device']

    # get definition of sde
    drift,diffusion = model_select(cfg_data['model'],cfg_data['frequency'])

    # generate dataset   
    dt = cfg_data['dt'] # 生成数据的dt
    t = torch.linspace(0,cfg_data['T'],cfg_data['nt']).cuda()# 生成数据的时间轴
    dim = cfg_data['dim'] #问题维数
    samples = cfg_data['sample'] #生成轨道数量
    dataset = DataSet(t, dt=dt, samples_num=samples, dim=dim, drift_fun=drift, diffusion_fun=diffusion,
                      initialization=torch.normal(mean=0., std=cfg_train['sigma_init'], size=(samples, dim)))
    data = dataset.get_data(plot_hist=True) # t, sample_num ,dim
    #np.save('data_2d.npy',data.detach().cpu().numpy())
    print("data: ", data.shape, data.max(), data.min())
    
    Data = TensorDataset(data.unsqueeze(0))

    train_loader = tud.DataLoader(dataset = Data, batch_size = 2, shuffle = False)
    val_loader = tud.DataLoader(dataset = Data, batch_size = 2, shuffle = False)

    # get NN
    if cfg_train["testfunction"]=="bump":
        testFunc = Bump
    elif cfg_train["testfunction"]=="gauss":
        testFunc = Gaussian
    net_drift , net_diffusion = net_select(cfg_nn)
    model = Model_pl(t = t, 
                     data = data, 
                     testFunc = testFunc, 
                     drift = drift, 
                     diffusion = diffusion, 
                     net_drift = net_drift,
                     net_diffusion = net_diffusion,  
                     cfg_train = cfg_train, 
                     cfg_nn = cfg_nn, 
                     device = data.device,
                     seed = config['seed'])


    # define callback
    if c_proj['checkpoint'] == False:
        save_file = os.path.join(c_proj["PATH"], 
                                c_proj["save_dir"]+cfg_nn["mode"])
        checkpoint_callback = ModelCheckpoint(                                
                                dirpath=save_file,
                                every_n_epochs = 1,
                                save_last = True,
                                monitor = 'val_loss',
                                mode = 'min',
                                save_top_k = c_proj['save_top_k'],
                                filename="model-{epoch:03d}-{val_loss:.4f}",
                            )
    if os.path.exists(save_file):
        print(f"The model directory exists. Overwrite? {c_proj['erase']}")
        if c_proj['erase'] == True:
            shutil.rmtree(save_file)
    max_epochs = cfg_train["epochs"] 
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # initiate trainer
    if c_proj['devices'] == 1 :
        trainer = pl.Trainer(max_epochs=max_epochs,
                            accelerator=c_proj['accelerator'], 
                            devices = c_proj['devices'],
                            callbacks = [checkpoint_callback,lr_monitor],
                            check_val_every_n_epoch=cfg_train['checkepoch'],
                            logger = wandb_logger
                            )#,
                            #strategy = 'deepspeed',gradient_clip_val=0.8)  # dp ddp deepspeed
    else: 
        device_num = [i for i in range(c_proj['devices'])]
        trainer = pl.Trainer(max_epochs=max_epochs,
                            accelerator=c_proj['accelerator'], 
                            devices = device_num,
                            callbacks = [checkpoint_callback,lr_monitor],
                            strategy = 'deepspeed',gradient_clip_val=0.8)
    trainer.fit(model, train_loader, val_loader)
    # if c_proj['save'] == True:
    #     save_path = os.path.join(c_proj['save_path'], 'model.pt')
    #     torch.save(model, save_path)
    #     print('save model done')
    # model.model.plot_error()
    
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config_file = getcfg()
    main(config_file)