import torch
import torch.nn as nn
import numpy as np
from torch import optim
import time
from utils.utils import set_seed
import utils
import itertools
import logging
logger = logging.getLogger(__name__)
# from postprocess import compute_error1D,plot_drift1D
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
from torch import optim
from utils.sampling import sampleTestFunc_all
from utils.loss import LpLoss, RRMSE, compute_error
from utils.plot import plot_error
from utils.functions import diffusion_knownnet,diffusion_knownnet_nonconst
from kan import KAN, LBFGS

class Model(nn.Module):
    def __init__(self, t, data, testFunc, drift, diffusion, net_drift,net_diffusion, cfg_train, cfg_nn, device):
        super(Model, self).__init__()
        '''
        t: observation time 
        t_number: obseravation snapshot number
        data: observation point cloud data
        device: cpu or cuda
        testFunc: test function
        net_drift: NN parameterization of drift function
        net_diffusion: NN parametrization of diffusion function
        ob_size: observation trajectory num
        sgd_ration: 
        plot_path: path to plot the visualization result
        dimension: dimension 
        variance_min/max: the variance interval of sampling test function
        gauss_num: total gauss sampling num
        td_type: temporal difference scheme
        gauss_samp_way: sampling method
        sgd_ratio: If one use sgd training, the ratio of one sgd batch of total train sample
        train_ratio:If one use early stopping, the train ratio of total sample
        error_type: how to calculate error
        plot_type: how to log the result 
        '''
        self.t = t #观测数据的时间轴
        self.t_number = len(t)
        self.data = data.to(device) #观测数据 (nt,n_sample,dim)
        self.device = device
        self.testFunc = testFunc #测试函数，Gaussian
        self.cfg_train = cfg_train
        self.cfg_nn = cfg_nn
        self.net_drift = net_drift.to(device) #表示漂移项等神经网络
        if cfg_train["known_diffusion"]:
            self.net_diffusion = diffusion_knownnet_nonconst
        else: 
            self.net_diffusion = net_diffusion.to(device)
        self.ob_size = data.shape[1] # 轨道数量
        self.sgd_ration = 0.8 #
        self.dimension = self.data.shape[-1]
        self.td_type =cfg_train["LMM"]
        self.drift = drift
        self.diffusion = diffusion
        self.mean_samp_way = cfg_train["mean_samp_way"] 
        self.var_samp_way = cfg_train["var_samp_way"] 
        self.sgd_ratio = cfg_train["sgd_ratio"]
        self.train_ratio = 0.8
        self.samp_coef = cfg_train['samp_coef']
        self.error_type = cfg_train['error_type']
        self.plot_type = cfg_train['plot_type']
        self.testfunname = cfg_train["testfunction"]
        if self.testfunname == "gauss":
            self.test_num = cfg_train["gauss_number"]
            self.sample_gaussian()
        else :
            self.test_num = cfg_train["bump_number"]
            self.generate_bump()
        
    def generate_bump(self):
        reg = self.data.max()-self.data.min()
        #start = self.data.min()+reg/(self.test_num+1)
        self.r = torch.ones(self.test_num ** self.dimension).to(self.device) * 10 * reg/(self.test_num+1)
        index = np.arange(self.data.shape[1]) 
        self.mu_list = self.data[:self.data.shape[0]-2,index[:self.test_num],:].permute(1,0,2) #samp_number,n_t,dim
        self.mu_list = self.mu_list + torch.randn_like(self.mu_list)*0.02
        #self.mu = torch.zeros([self.test_num ** self.dimension,self.data.shape[0]-2,self.dimension]).double().to(self.device)
        #for i in range(self.mu.shape[0]):
        #    for j in range(self.data.shape[0]-2):
        #        for k in range(self.dimension):
        #            self.mu[i,j,k]= start + reg*torch.tensor((i//self.test_num**k)%self.test_num).double()/(self.test_num+1)
        # print("mu",self.mu_list.shape)
        testn = self.testFunc(self.mu_list, self.r, self.device)
        testnp1 = self.testFunc(self.mu_list, self.r, self.device)
        testnp2 = self.testFunc(self.mu_list, self.r, self.device)
        TX = self.data  # [t, sample, dim]

        self.testn_0 = testn(TX[:-2], diff_order=0) #bump,t-2,sample
        self.testn_1 = testn(TX[:-2], diff_order=1) #bump,t-2,sample,dim
        self.testn_2 = testn(TX[:-2], diff_order=2) #bump,t-2,sample,dim,dim 
        # print("bumpshape:",self.testn_2.shape) 

        self.testnp1_0 = testnp1(TX[1:-1], diff_order=0) #bump,t-2,sample
        self.testnp1_1 = testnp1(TX[1:-1], diff_order=1) #bump,t-2,sample,dim
        self.testnp1_2 = testnp1(TX[1:-1], diff_order=2) #bump,t-2,sample,dim,dim

        self.testnp2_0 = testnp2(TX[2:], diff_order=0) #bump,t-2,sample
        self.testnp2_1 = testnp2(TX[2:], diff_order=1) #bump,t-2,sample,dim
        self.testnp2_2 = testnp2(TX[2:], diff_order=2) #bump,t-2,sample,dim,dim

        self.data_test = self.data.reshape(-1,self.dimension)
        index_test = np.arange(self.data_test.shape[0])
        np.random.shuffle(index_test)
        self.data_test = self.data_test[index_test[:10000],:]

    def sample_gaussian(self):
        self.mu_list_all, self.variance = sampleTestFunc_all(data = self.data, 
                           samp_number = self.test_num, 
                           mean_samp_way = self.mean_samp_way, 
                           var_samp_way = self.var_samp_way, 
                           samp_coef = self.samp_coef,
                           device = self.device)
        self.sampleTestFunc()
        self.data_test = self.data.reshape(-1,self.dimension)
        index_test = np.arange(self.data_test.shape[0])
        np.random.shuffle(index_test)
        self.data_test = self.data_test[index_test[:10000],:]

    def sampleTestFunc(self):
        '''
        Compute the Gaussian and its derivative for the data point
        '''
        # for i in range(self.sampling_number):
            
        testn = self.testFunc(self.mu_list_all, self.variance, self.device)
        testnp1 = self.testFunc(self.mu_list_all, self.variance, self.device)
        testnp2 = self.testFunc(self.mu_list_all, self.variance, self.device)
        TX = self.data  # [t, sample, dim]

        self.testn_0 = testn(TX[:-2], diff_order=0) #gauss,t-2,sample
        self.testn_1 = testn(TX[:-2], diff_order=1) #gauss,t-2,sample,dim
        self.testn_2 = testn(TX[:-2], diff_order=2) #gauss,t-2,sample,dim,dim 
        # print("gaussshape:",self.testn_2.shape)  

        self.testnp1_0 = testnp1(TX[1:-1], diff_order=0) #gauss,t-2,sample
        self.testnp1_1 = testnp1(TX[1:-1], diff_order=1) #gauss,t-2,sample,dim
        self.testnp1_2 = testnp1(TX[1:-1], diff_order=2) #gauss,t-2,sample,dim,dim

        self.testnp2_0 = testnp2(TX[2:], diff_order=0) #gauss,t-2,sample
        self.testnp2_1 = testnp2(TX[2:], diff_order=1) #gauss,t-2,sample,dim
        self.testnp2_2 = testnp2(TX[2:], diff_order=2) #gauss,t-2,sample,dim,dim
    
    def compute_loss_exact(self,TX, mode = 'sgd'):
        '''
        Compute the residual loss for each frame
        '''

        drift_pred = self.drift(TX)   # [t, sample, dim]
        diffusion_pred = self.diffusion(TX).squeeze(-1) # [t,sample, dim]
        # gauss: gauss,t,sample
        # gauss1: gauss,t,sample,dim
        # gauss2: gauss,t,sample,dim,dim
        # drift_pred: t,sample,dim

        index_all = np.arange(self.test_num)
        if mode == 'train':
            index = index_all[:int(self.test_num)]
        elif mode == 'sgd':
            n = int(self.train_ratio*self.test_num)
            index = np.arange(n)
            np.random.shuffle(index)
            index = index[:int(self.sgd_ratio*n)]
        elif mode == 'val':
            n = int(self.train_ratio*self.test_num)
            index = np.arange(n)
            index = index[int(self.sgd_ratio*n):]
        


        # Compute the (mu,\grad Gauss)
        # drift:t-2,sample,dim -> 1,t-2,sample,dim
        # testn_1: gauss,t-2,sample,dim
        # mean: gauss*(t-2)
        An_1 = torch.mean(torch.sum(drift_pred[:-2,:,:].unsqueeze(0) * self.testn_1[index], dim=3), dim=2).view(-1) #gauss*t-2
        # Compute the (sigma,Dij Gauss)
        # self.D: sample,dim
        # testn_2: gauss,t-2,sample,dim
        # mean: gauss*(t-2)
        self.D = 1/2*(diffusion_pred)#  [t,sample, dim]
        
        An_2 = torch.mean(torch.einsum("ijkl,jkl->ijk", [self.testn_2[index], self.D[:-2,:,:]]), dim=2).view(-1) ##gauss*t

        An = An_1 + An_2

        Anp1_1 = torch.mean(torch.sum(drift_pred[1:-1,:,:].unsqueeze(0) * self.testnp1_1[index], dim=3), dim=2).view(-1)
        Anp1_2 = torch.mean(torch.einsum("ijkl,jkl->ijk", [self.testnp1_2[index], self.D[1:-1,:,:]]), dim=2).view(-1)
        Anp1 = Anp1_1 + Anp1_2

        Anp2_1 = torch.mean(torch.sum(drift_pred[2:,:,:].unsqueeze(0) * self.testnp2_1[index], dim=3), dim=2).view(-1)
        Anp2_2 = torch.mean(torch.einsum("ijkl,jkl->ijk", [self.testnp2_2[index], self.D[2:,:,:]]), dim=2).view(-1)
        Anp2 = Anp2_1 + Anp2_2
        

        # taking mean in the dimension of sample 
        rbn = torch.mean(self.testn_0[index], dim=2).view(-1) #gauss*t
        #rbnp1 = torch.mean(self.testnp1_0[index], dim=2).view(-1) #gauss*t
        rbnp2 = torch.mean(self.testnp2_0[index], dim=2).view(-1) #gauss*t
        
        
        dt = (torch.max(self.t)-torch.min(self.t)) / (self.t_number - 1)
        
        if self.td_type == 'LMM_3':
            Aq = (An + 4*Anp1 + Anp2) / 3 * dt
            bq = rbnp2 - rbn
        residual = Aq - bq
        #self.loss_buffer.append(self.loss)
        return residual

    def compute_loss(self,TX, mode = 'sgd'):
        '''
        Compute the residual loss for each frame
        '''

        drift_pred = self.net_drift(TX)   # [t, sample, dim]
        diffusion_pred = self.net_diffusion(TX) # [t,sample, dim]
        # gauss: gauss,t,sample
        # gauss1: gauss,t,sample,dim
        # gauss2: gauss,t,sample,dim,dim
        # drift_pred: t,sample.dim

        index_all = np.arange(self.test_num)
        if mode == 'train':
            index = index_all[:int(self.test_num)]
        elif mode == 'sgd':
            n = int(self.train_ratio*self.test_num)
            index = np.arange(n)
            np.random.shuffle(index)
            index = index[:int(self.sgd_ratio*n)]
        elif mode == 'val':
            n = int(self.train_ratio*self.test_num)
            index = np.arange(n)
            index = index[int(self.sgd_ratio*n):]
        


        # Compute the (mu,\grad Gauss)
        # drift:t-2,sample,dim -> 1,t-2,sample,dim
        # testn_1: gauss,t-2,sample,dim
        # mean: gauss*(t-2)
        An_1 = torch.mean(torch.sum(drift_pred[:-2,:,:].unsqueeze(0) * self.testn_1[index], dim=3), dim=2).view(-1) #gauss*t-2
        # Compute the (sigma,Dij Gauss)
        # self.D: sample,dim
        # testn_2: gauss,t-2,sample,dim
        # mean: gauss*(t-2)
        self.D = 1/2*(diffusion_pred)#  [t,sample, dim]
        
        An_2 = torch.mean(torch.einsum("ijkl,jkl->ijk", [self.testn_2[index], self.D[:-2,:,:]]), dim=2).view(-1) ##gauss*t

        An = An_1 + An_2

        Anp1_1 = torch.mean(torch.sum(drift_pred[1:-1,:,:].unsqueeze(0) * self.testnp1_1[index], dim=3), dim=2).view(-1)
        Anp1_2 = torch.mean(torch.einsum("ijkl,jkl->ijk", [self.testnp1_2[index], self.D[1:-1,:,:]]), dim=2).view(-1)
        Anp1 = Anp1_1 + Anp1_2

        Anp2_1 = torch.mean(torch.sum(drift_pred[2:,:,:].unsqueeze(0) * self.testnp2_1[index], dim=3), dim=2).view(-1)
        Anp2_2 = torch.mean(torch.einsum("ijkl,jkl->ijk", [self.testnp2_2[index], self.D[2:,:,:]]), dim=2).view(-1)
        Anp2 = Anp2_1 + Anp2_2
        

        # taking mean in the dimension of sample 
        rbn = torch.mean(self.testn_0[index], dim=2).view(-1) #gauss*t
        #rbnp1 = torch.mean(self.testnp1_0[index], dim=2).view(-1) #gauss*t
        rbnp2 = torch.mean(self.testnp2_0[index], dim=2).view(-1) #gauss*t
        
        
        dt = (torch.max(self.t)-torch.min(self.t)) / (self.t_number - 1)
        
        if self.td_type == 'LMM_3':
            Aq = (An + 4*Anp1 + Anp2) / 3 * dt
            bq = rbnp2 - rbn
        residual = Aq - bq
        #self.loss_buffer.append(self.loss)
        return residual

    def compute_error(self):
        with torch.no_grad():
            drift_error, diffusion_error = compute_error(drift = self.drift,
                                                         net_drift=self.net_drift,
                                                         diffusion=self.diffusion,
                                                         net_diffusion=self.net_diffusion,
                                                         device=self.device,
                                                         data=self.data,
                                                         error_type=self.error_type,
                                                         )
        return drift_error, diffusion_error
    def plot_error(self):
        with torch.no_grad():
            plot_error(drift_fun = self.drift,
                       drift_net = self.net_drift,
                       diffusion_fun = self.diffusion,
                       diffusion_net = self.net_diffusion,
                       device = self.device,
                       plot_type = self.plot_type)
    

class Model_pl(pl.LightningModule):
    def __init__(self, t, data, testFunc, drift, diffusion, net_drift,net_diffusion, cfg_train, cfg_nn, seed, device):
        super(Model_pl, self).__init__()
        self.model = Model( t = t, 
                           data = data, 
                           testFunc = testFunc, 
                           drift = drift, 
                           diffusion = diffusion, 
                           net_drift = net_drift,
                           net_diffusion = net_diffusion, 
                           cfg_train = cfg_train, 
                           cfg_nn = cfg_nn, 
                           device = device)
        self.lr = cfg_nn['lr'] 
        self.mode = cfg_nn['mode']
        self.step_size = cfg_nn['step_size'] 
        self.weight_decay = cfg_nn['weight_decay'] 
        self.gamma = cfg_nn['gamma'] 
        self.criterion = nn.MSELoss()#H1_loss(alpha = 0.1)
        self.criterion_val = nn.MSELoss()
        self.plotstep = cfg_train['plotstep']
        self.seed = seed
        
        
    def forward(self, x,mode):
        loss = self.model.compute_loss(x,mode)
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx):  
         
        residual = self(batch[0][0],'sgd')
        #res_exact = self.model.compute_loss_exact(batch[0][0],'sgd')
        #print(out.shape,y.shape)
        #loss,l2, l_phy = self.criterion(out,y)#torch.mean(torch.abs(out.view(batch_size,-1)-10*y.view(batch_size,-1)) ** 2)
        loss = torch.sum(residual**2)
        #loss_exact = torch.sum(res_exact**2)
        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        #self.log("loss_exact", loss_exact, on_epoch=True, prog_bar=True, logger=True)
        
        # self.log("loss_l2", l2, on_epoch=True, prog_bar=True, logger=True)
        # self.log("l_phy", l_phy, on_epoch=True, prog_bar=True, logger=True)
        #wandb.log({"loss": loss.item(),'loss_data':l2.item(),'loss_phy':l_phy.item()})
        #wandb.log({"loss": loss.item()})
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx):
        if self.global_step == 0:
            wandb.define_metric("drift_error", summary="min")
        if self.global_step in self.plotstep:
            self.model.plot_error()
        
         
        residual_val = self(val_batch[0][0],'val')
        drift_error, diffusion_error = self.model.compute_error()
       
        val_residual = torch.sum(residual_val**2)

        val_loss = drift_error  + diffusion_error

        self.log('val_residual', val_residual, on_epoch=True, prog_bar=True, logger=True)
        #wandb.log({'val_residual': val_residual.item()})
        self.log('drift_error', drift_error, on_epoch=True, prog_bar=True, logger=True)
        #wandb.log({'drift_error': drift_error.item()})
        self.log('diffusion_error', diffusion_error, on_epoch=True, prog_bar=True, logger=True)
        #wandb.log({'diffusion_error': diffusion_error.item()})
        
        self.log('val_loss',val_loss, on_epoch=True, prog_bar=True, logger=True)

        return val_loss
    

    def configure_optimizers(self, optimizer=None, scheduler=None):
            if optimizer is None:
                optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            if  scheduler is None:
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
                #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = self.step_size, eta_min= self.eta_min)
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            }
        }
    