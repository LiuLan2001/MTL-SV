import numpy as np
import torch
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

class CoordNet(nn.Module):
    def __init__(self, in_features, out_features, init_features=64,num_res = 10):
        super(CoordNet,self).__init__()

        self.num_res = num_res

        self.net = []

        self.net.append(ResBlock(in_features,init_features))
        self.net.append(ResBlock(init_features,2*init_features))
        self.net.append(ResBlock(2*init_features,4*init_features))

        for i in range(self.num_res):
            self.net.append(ResBlock(4*init_features,4*init_features))
        self.net = nn.Sequential(*self.net)
        
        self.fc1 = ResBlock(4*init_features, out_features)
        self.fc2 = ResBlock(4*init_features, out_features)
        self.fc3 = ResBlock(4*init_features, out_features)

        self.n_output_dims = out_features

    def forward(self, coords):
        output = self.net(coords)
        out1 = self.fc1(output)
        out2 = self.fc2(output)
        out3 = self.fc3(output)
        out = (out1+out2+out3)/3
        data = torch.stack([out1,out2,out3], axis=-1)
        var = torch.var(data, axis=-1)
        return out, var, out1, out2, out3

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

def kl_d(var, mse):
    mse = mse/mse.sum().detach()
    var = var/var.sum()
    kl_loss = torch.nn.functional.kl_div(
        torch.log(var+1.e-16),
        torch.log(mse+1.e-16),
        reduction='none',
        log_target=True,
    ).mean()
    return kl_loss

def trainNet(model,args,dataset):
    result_dir = os.path.join(args.result_dir, f'{args.dataset}', f'MDSRN')

    logs_dir = os.path.join(result_dir, 'logs')
    checkpoints_dir = os.path.join(result_dir, 'checkpoints')
    outputs_dir = os.path.join(result_dir, 'outputs')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    loss_log_file = result_dir+'/'+'loss-'+'-'+str(args.interval)+'-'+str(args.init)+'-'+str(args.active)+'.txt'
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=1e-6, fused=True)
    mse_loss = nn.MSELoss()
    scaler = GradScaler(enabled=args.fp16)
    
    t = 0
    start_time = time.time()
    with open(loss_log_file,"a") as f:
        f.write(f"time:{time.time()}")
        f.write('\n')
    for epoch in range(1,args.num_epochs+1):
        model.train()
        training_data_inputs, training_data_outputs, batchIndexGenerator = dataset.get_data()
        loss_mse = 0
        loss_kl = 0
        loss_grad = 0
        loop = tqdm.tqdm(batchIndexGenerator)
        l = 30*(500**((epoch-1) / (args.num_epochs-1))-1)/499 

        for current_idx, next_idx in loop:
            coord = training_data_inputs[current_idx:next_idx].contiguous()
            v = training_data_outputs[current_idx:next_idx].contiguous()
            
            optimizer.zero_grad()
            with autocast(enabled=args.fp16):
                mean, var, out1, out2, out3 = model(coord)
                mse = (mse_loss(out1.view(-1),v.view(-1)) + mse_loss(out2.view(-1),v.view(-1)) + mse_loss(out3.view(-1),v.view(-1))) / 3
                kl = kl_d(var.view(-1), ((mean.view(-1)-v.view(-1))**2))
                loss = mse + l*kl

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_mse += mse.mean().item()
            loss_kl += kl.mean().item()

            loop.set_description(f'Epoch [{epoch}/{args.num_epochs}]')
            loop.set_postfix(mse=loss_mse, kl=loss_kl)
        adjust_lr(args, optimizer, epoch)

        with open(loss_log_file,"a") as f:
            f.write(f"Epochs {epoch}: loss = {loss_mse}, lr = {optimizer.param_groups[0]['lr']}")
            f.write('\n')

        if epoch%args.checkpoint == 0 or epoch==1:
            torch.save(model.state_dict(),checkpoints_dir+'/'+'-'+str(args.interval)+'-'+str(args.init)+'-'+str(epoch)+'.pth')
            
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
                    dat, var, out1, out2, out3 = model(coord)
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