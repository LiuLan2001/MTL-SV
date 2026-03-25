import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils import BatchIndex,get_mgrid,fast_random_choice,count_params,cleanup,seed_everything,adjust_lr

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class ResBlock(nn.Module):
    def __init__(self,in_features,out_features,nonlinearity='relu'):
        super(ResBlock,self).__init__()

        self.net = []

        self.net.append(SineLayer(in_features,out_features))

        self.net.append(SineLayer(out_features,out_features))

        self.flag = (in_features!=out_features)

        if self.flag:
            self.transform = SineLayer(in_features,out_features)

        self.net = nn.Sequential(*self.net)
    
    def forward(self,features):
        outputs = self.net(features)
        if self.flag:
            features = self.transform(features)
        return 0.5*(outputs+features)
    
# The following presents two versions of the model architecture. The upper one is the original version, 
# while the lower one is the improved version. For specific details, please refer to Appendix 
# "COMPARISON OF DIFFERENT MODEL ARCHITECTURES AND THE LIMITATION OF OUR METHOD".
    
class CoordNet(nn.Module):
    def __init__(self, in_features, out_features, init_features,num_res):
        super(CoordNet,self).__init__()

        self.num_res = num_res

        self.net1 = []
        self.net1.append(ResBlock(in_features,init_features))
        self.net1.append(ResBlock(init_features,2*init_features))
        self.net1.append(ResBlock(2*init_features,4*init_features))
        for i in range(self.num_res):
            self.net1.append(ResBlock(4*init_features,4*init_features))
        self.net1 = nn.Sequential(*self.net1)

        self.net2 = []
        self.net2.append(ResBlock(in_features,init_features))
        self.net2.append(ResBlock(init_features,2*init_features))
        self.net2.append(ResBlock(2*init_features,4*init_features))
        for i in range(self.num_res):
            self.net2.append(ResBlock(4*init_features,4*init_features))
        self.net2 = nn.Sequential(*self.net2)
        
        self.gate1 = nn.Linear(in_features, 3, bias=False)
        self.gate2 = nn.Linear(in_features, 3, bias=False)

        self.fc1 = ResBlock(4*init_features, out_features)
        self.fc2 = ResBlock(4*init_features, out_features)

    def forward(self, coords):
        output1 = self.net1(coords)
        output2 = self.net2(coords)
        
        g1 = F.softmax(self.gate1(coords), dim=1)
        g2 = F.softmax(self.gate2(coords), dim=1)

        out = self.fc1(g1[:, 0:1] * output1 + g1[:, 1:2] * output2)
        var = self.fc2(g2[:, 0:1] * output1 + g2[:, 1:2] * output2)
        var = F.softplus(var)
        return out, var

# class CoordNet(nn.Module):
#     def __init__(self, in_features, out_features, init_features,num_res):
#         super(CoordNet,self).__init__()

#         self.num_res = num_res

#         self.net1 = []
#         self.net1.append(ResBlock(in_features,init_features))
#         self.net1.append(ResBlock(init_features,2*init_features))
#         self.net1.append(ResBlock(2*init_features,4*init_features))
#         for i in range(self.num_res):
#             self.net1.append(ResBlock(4*init_features,4*init_features))
#         self.net1 = nn.Sequential(*self.net1)

#         self.net2 = []
#         self.net2.append(ResBlock(in_features,init_features))
#         self.net2.append(ResBlock(init_features,2*init_features))
#         self.net2.append(ResBlock(2*init_features,4*init_features))
#         for i in range(self.num_res):
#             self.net2.append(ResBlock(4*init_features,4*init_features))
#         self.net2 = nn.Sequential(*self.net2)

#         self.gate1 = nn.Linear(in_features, 3, bias=False)
#         self.gate2 = nn.Linear(in_features, 3, bias=False)

#         self.fc1 = ResBlock(4*init_features, out_features)
#         self.fc2 = ResBlock(4*init_features, out_features*2)

#     def forward(self, coords):
#         output1 = self.net1(coords)
#         output2 = self.net2(coords)
        
#         g1 = F.softmax(self.gate1(coords), dim=1)
#         g2 = F.softmax(self.gate2(coords), dim=1)

#         out = self.fc1(g1[:, 0:1] * output1 + g1[:, 1:2] * output2)
#         var_ten = self.fc2(g2[:, 0:1] * output1 + g2[:, 1:2] * output2) 
#         var = var_ten[:,0] - var_ten[:,1]
#         var = F.relu(var)
#         return out, var

import torch
from torch import nn
import os
import numpy as np
import torch.optim as optim
import tqdm
from datetime import datetime
from shutil import copy, copytree
import json
import time
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity
import math

class CorrLoss(nn.Module):
    def __init__(self):
        super(CorrLoss,self).__init__()
    def forward(self,y_true, y_pred):
        num = torch.mean((y_true -torch.mean(y_true))*(y_pred - torch.mean(y_pred)))
        den = torch.std(y_true)* torch.std(y_pred) + 1e-8
        correlation =num/den
        return 1-correlation
    
def trainNet(model,args,dataset):
    result_dir = os.path.join(args.result_dir, f'{args.dataset}', f'MMOE')

    logs_dir = os.path.join(result_dir, 'logs')
    checkpoints_dir = os.path.join(result_dir, 'checkpoints')
    outputs_dir = os.path.join(result_dir, 'outputs')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    loss_log_file = result_dir+'/'+'loss-'+'-'+str(args.interval)+'-'+'-'+str(args.active)+'.txt'
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=1e-6, fused=True)
    mse_loss = nn.MSELoss()
    corr_loss = CorrLoss()
    scaler = GradScaler(enabled=args.fp16)

    t = 0
    start_time = time.time()
    corr_losses = []
    mse_losses = []
    with open(loss_log_file,"a") as f:
        f.write(f"time:{time.time()}")
        f.write('\n')
    for epoch in range(1,args.num_epochs+1):
        model.train()
        training_data_inputs, training_data_outputs, batchIndexGenerator = dataset.get_data()
        loss_mse = 0
        loss_corr = 0
        loop = tqdm.tqdm(batchIndexGenerator)
        
        for current_idx, next_idx in loop:
            coord = training_data_inputs[current_idx:next_idx].contiguous()
            v = training_data_outputs[current_idx:next_idx].contiguous()

            optimizer.zero_grad()
            with autocast(enabled=args.fp16):
                mean, var = model(coord)
                var = var.view(-1) + 1.e-9

                mse = mse_loss(mean.view(-1),v.view(-1)) 
                corr = corr_loss(var.view(-1), ((mean.view(-1)-v.view(-1))**2))
                if epoch>2:
                    mu_corr = sum(corr_losses) / len(corr_losses)
                    mu_mse = sum(mse_losses) / len(mse_losses)

                    sigma_corr = torch.std(torch.tensor(corr_losses))
                    sigma_mse = torch.std(torch.tensor(mse_losses))
                    a_corr = sigma_corr / mu_corr
                    a_mse = sigma_mse / mu_mse
                    loss = a_mse*mse + a_corr*0.0008*corr  
                    #For the original version, we used 0.0008.For the improved version, we used 0.0007.
                    
                else:
                    a_corr=1
                    a_mse=1
                    loss =  mse+0.0001*(corr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_mse += mse.mean().item()
            loss_corr += corr.mean().item()

            loop.set_description(f'Epoch [{epoch}/{args.num_epochs}]')
            loop.set_postfix(mse=loss_mse, corr=loss_corr)
        adjust_lr(args, optimizer, epoch)
        mse_losses.append(loss_mse)
        corr_losses.append(loss_corr)
        # scheduler.step()

        with open(loss_log_file,"a") as f:
            f.write(f"{epoch} {loss_mse} {loss_corr} {a_corr} {a_mse}")
            f.write('\n')

        if epoch%args.checkpoint == 0 or epoch==1:
            torch.save(model.state_dict(),checkpoints_dir+'/'+'-'+str(args.interval)+'-'+'-'+str(epoch)+'.pth')
    with open(loss_log_file,"a") as f:
        f.write(f"time:{time.time()}")
        f.write(f"time:{time.time()-start_time}")
        f.write('\n')

@torch.no_grad()
def inf(model,dataset,args, result_dir=None):
    ckpt = './'+args.dataset+args.ckpt
    result_dir = os.path.dirname(os.path.dirname(ckpt)) if result_dir is None else result_dir
    outputs_dir = os.path.join(result_dir, 'outputs', 'inference')
    var_dir = os.path.join(result_dir, 'outputs', 'var')
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(var_dir, exist_ok=True)

    model.eval()
    samples = dataset.samples
    for i in range(len(samples)):  
        for j in range(0,dataset.interval+1):
            frame_idx = samples[i] + j
            val_data_inputs, batchIndexGenerator =dataset._get_testing_data(frame_idx)
            v = []
            d = []
            loop = tqdm.tqdm(batchIndexGenerator)
            for current_idx, next_idx in loop:
                coord = val_data_inputs[current_idx:next_idx]
                with torch.no_grad():
                    dat, var = model(coord)
                    dat = dat.view(-1)
                    var = var.view(-1)
                    d.append(dat)
                    v.append(var)
            d = torch.cat(d,dim=-1).float()
            d = d.detach().cpu().numpy()
            d = np.asarray(d,dtype='<f')
            out_path = f'{outputs_dir}/{frame_idx:04}.dat'
            d.tofile(out_path, format='<f')
            v = torch.cat(v,dim=-1).float()
            v = v.detach().cpu().numpy()
            v = np.asarray(v,dtype='<f')
            var_path = f'{var_dir}/{frame_idx:04}.dat'
            v.tofile(var_path, format='<f')