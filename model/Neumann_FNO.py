import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import scipy.io as scio
import time
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
from torch import optim

class DC1(nn.Module):
    def __init__(self, inc, interc, outc, k_size):
        super(DC1, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels = inc, out_channels = interc, kernel_size = k_size, stride = 1, padding = (k_size-1)//2, dilation = 1)
        self.cnn2 = nn.Conv2d(in_channels = interc, out_channels = outc, kernel_size = k_size, stride = 1, padding = (k_size-1)//2, dilation = 1)
        self.activation = F.gelu

    def forward(self, x):
        x = self.cnn1(x)
        x = self.activation(x)
        x = self.cnn2(x)
        # x = self.activation(x)
        return x
    

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, cin):
        super(FNO2d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.cin = cin
        self.padding = 9 # pad the domain if input is non-periodic
        self.activation = F.gelu
        
        self.fc0 = nn.Linear(self.cin + 2, self.width) # input is (cin, x, y)
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2)
        
        self.DC = DC1(self.cin, 8, self.cin, 3)

    def forward(self, x):
        x = self.DC(x)
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        # x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = self.activation(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
    def get_grid(self, shape, device):
        batchsize, channel, size_x, size_y = shape[0], shape[1], shape[2], shape[3]
        x = np.linspace(0, 1, size_x)
        y = np.linspace(0, 1, size_y)
        X, Y = np.meshgrid(x, y)
        X = torch.tensor(X, dtype=torch.float).to(device)
        Y = torch.tensor(Y, dtype=torch.float).to(device)
        gridx = X.reshape(1, 1, size_x, size_y).repeat([batchsize, 1, 1, 1])
        gridy = Y.reshape(1, 1, size_x, size_y).repeat([batchsize, 1, 1, 1])
        return torch.cat((gridx, gridy), dim=1)







class Neumann_FNO(nn.Module):
    def __init__(self, N, k, modes1, modes2, width):
        super(Neumann_FNO, self).__init__()
        
        self.k = k
        self.N = N
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        self.fno1 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, cin = 1)
        self.fno2 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, cin = 2)
        self.fno3 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, cin = 2)
        #self.fno4 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, cin = 2)
        #self.fno5 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, cin = 2)
        # self.fno6 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, cin = 2)

    def forward(self, x):
        q = x[:,0,:,:]
        f = x[:,1,:,:]
        f = f.unsqueeze(1)
        u0 = self.fno1(f)
        
        q = q.unsqueeze(1)
        u1 = self.fno2(-self.k**2*q*u0)

        u2 = self.fno3(-self.k**2*q*u1)
        #u3 = self.fno4(-self.k**2*q*u2)
        #u4 = self.fno5(-self.k**2*q*u3)
        # u5 = self.fno6(-self.k**2*q*u4)
        u = u0 + u1 + u2 #+ u3 + u4 #+ u5
        return u

class Neumann_FNO_res(nn.Module):
    def __init__(self, N, k, modes1, modes2, width):
        super(Neumann_FNO_res, self).__init__()
        
        self.k = k
        self.N = N
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        self.fno1 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, cin = 3)
        self.fno2 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, cin = 2)
        self.fno3 = FNO2d(modes1 = self.modes1, modes2 = self.modes2, width = self.width, cin = 2)
    
    def forward(self, x):
        q = x[:,0,:,:]
        f = x[:,1:,:,:]#.permute(0,2,3,1)
        #f = f.unsqueeze(1)
        
        u0 = self.fno1(f)
        
        q = q.unsqueeze(1)
        u1 = self.fno2(-self.k**2*q*u0)

        u2 = self.fno3(-self.k**2*q*u1)
        
        u = u0 + u1 + u2
        return u



class NS_FNO(pl.LightningModule):
    def __init__(self, N, 
                        k, 
                        modes1, 
                        modes2, 
                        width,  
                        lr = 0.001, 
                        step_size= 100, 
                        gamma= 0.5, 
                        weight_decay= 1e-4):
        super(NS_FNO, self).__init__()
        self.model = Neumann_FNO(N, k, modes1, modes2, width)
        self.lr = lr 
        self.step_size = step_size
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.criterion = LossPDE()#H1_loss(alpha = 0.1)
        self.criterion_val = RRMSE()
        self.val_iter = 0
        self.N = N
        self.k = k
        
        
    def forward(self, x):
        u = self.model(x)
        return u

    def training_step(self, batch: torch.Tensor, batch_idx):    
        x, y = batch
        batch_size = x.shape[0]
        out = self(x)
        #print(out.shape,y.shape)
        #loss,l2, l_phy = self.criterion(out,y)#torch.mean(torch.abs(out.view(batch_size,-1)-10*y.view(batch_size,-1)) ** 2)
        loss = self.criterion(out,y,self.N, self.k,x)
        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # self.log("loss_l2", l2, on_epoch=True, prog_bar=True, logger=True)
        # self.log("l_phy", l_phy, on_epoch=True, prog_bar=True, logger=True)
        # wandb.log({"loss": loss.item(),'loss_data':l2.item(),'loss_phy':l_phy.item()})
        wandb.log({"loss": loss.item()})
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx):
        self.val_iter += 1
        x, y= val_batch
        batch_size = x.shape[0]
        #out = self(sos,src)+10*self.homo_field.unsqueeze(0) #new
        out = self(x)
        val_loss = self.criterion_val(out.view(batch_size,-1),y.view(batch_size,-1))
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"val_loss": val_loss.item()})
       
        if self.val_iter %10 ==0:
            #self.log_wandb_image(wandb,sos[0].detach().cpu(),(y-self.homo_field.unsqueeze(0))[0].detach().cpu(),(out-10*self.homo_field.unsqueeze(0))[0].detach().cpu())
            self.log_wandb_image(wandb,x[0,0].detach().cpu(),y[0,0].detach().cpu(),out[0,0].detach().cpu())
        return val_loss
    
    def log_wandb_image(self,wandb,  sos, field, pred_field):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax = ax.flatten()
        ax0 = ax[0].imshow(sos, cmap="inferno")
        ax[0].set_title("Sound speed")
        ax[1].imshow(field, cmap="RdBu_r")
        ax[1].set_title("Field")
        ax[2].imshow(pred_field, cmap="RdBu_r")
        ax[2].set_title("Predicted field")
        img = wandb.Image(plt)
        wandb.log({'Image': img})
        plt.close()

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
        },
    }

class NS_FNO_Residual(pl.LightningModule):
    def __init__(self, model_old,
                        N, 
                        k, 
                        modes1, 
                        modes2, 
                        width,  
                        lr = 0.001, 
                        step_size= 100, 
                        gamma= 0.5, 
                        weight_decay= 1e-4):
        super(NS_FNO_Residual, self).__init__()
        self.model_old = model_old
        self.model_old.freeze()
        #self.model = Neumann_FNO(N, k, modes1, modes2, width)
        self.model = Neumann_FNO_res(N, k, modes1, modes2, width)
        
        self.lr = lr 
        self.step_size = step_size
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.criterion = H1_loss(alpha = 1)#LpLoss()
        self.criterion_val = RRMSE()
        self.val_iter = 0
        
        
    def forward(self, x):
        u = self.model_old(x)
        input = torch.cat((x,u),dim = 1)
        out = self.model(input)
        return out, u

    def training_step(self, batch: torch.Tensor, batch_idx):    
        x, y = batch
        batch_size = x.shape[0]
        out, u = self(x)
        res = y - u
        #loss = self.criterion(out.view(batch_size,-1),res.view(batch_size,-1))#torch.mean(torch.abs(out.view(batch_size,-1)-10*y.view(batch_size,-1)) ** 2)
        loss,l2, l_phy = self.criterion(out,res)#torch.mean(torch.abs(out.view(batch_size,-1)-10*y.view(batch_size,-1)) ** 2)
        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss_l2", l2, on_epoch=True, prog_bar=True, logger=True)
        self.log("l_phy", l_phy, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"loss": loss.item(),'loss_data':l2.item(),'loss_phy':l_phy.item()})
        # self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # wandb.log({"loss": loss.item()})
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx):
        self.val_iter += 1
        x, y= val_batch
        batch_size = x.shape[0]
        
        out, u = self(x)
        out = u + out
        val_loss = self.criterion_val(out.view(batch_size,-1),y.view(batch_size,-1))
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        wandb.log({"val_loss": val_loss.item()})
       
        if self.val_iter %10 ==0:
            #self.log_wandb_image(wandb,sos[0].detach().cpu(),(y-self.homo_field.unsqueeze(0))[0].detach().cpu(),(out-10*self.homo_field.unsqueeze(0))[0].detach().cpu())
            self.log_wandb_image(wandb,x[0,0].detach().cpu(),y[0,0].detach().cpu(),out[0,0].detach().cpu())
        return val_loss
    
    def log_wandb_image(self,wandb,  sos, field, pred_field):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax = ax.flatten()
        ax0 = ax[0].imshow(sos, cmap="inferno")
        ax[0].set_title("Sound speed")
        ax[1].imshow(field, cmap="RdBu_r")
        ax[1].set_title("Field")
        ax[2].imshow(pred_field, cmap="RdBu_r")
        ax[2].set_title("Predicted field")
        img = wandb.Image(plt)
        wandb.log({'Image': img})
        plt.close()

    def configure_optimizers(self, optimizer=None, scheduler=None):
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if  scheduler is None:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
            #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = self.step_size, eta_min= self.eta_min)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler
        },
    }