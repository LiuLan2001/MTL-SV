import random
import numpy as np
import torch
import gc
from torch import optim
import math

class BatchIndex:
    def __init__(self, size, batch_size, shuffle=True):
        self.index_list = torch.as_tensor([(x, min(x + batch_size, size)) for x in range(0, size, batch_size)])
        
        if shuffle:
            self.index_list = self.index_list[torch.randperm(len(self.index_list))]
        
        self.pos = -1

    def __next__(self):
        self.pos += 1
        if self.pos >= len(self.index_list):
            raise StopIteration
        return self.index_list[self.pos]

    def __iter__(self):
        self.pos = -1
        return self

    def __len__(self):
        return len(self.index_list)
    
def get_mgrid(sidelen, dim=2, s=1,t=0):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]:s, :sidelen[1]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[0] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
    elif dim == 3:
        ranges = [
            torch.arange(0, sidelen[0], s, device='cuda:0'),
            torch.arange(0, sidelen[1], s, device='cuda:0'),
            torch.arange(0, sidelen[2], s, device='cuda:0'),
        ]
        grid = torch.meshgrid(ranges, indexing='ij')
        pixel_coords = torch.stack(grid, dim=-1)[None, ...].float()
        pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[0] - 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    elif dim == 4:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]:(t+1), :sidelen[1]:s, :sidelen[2]:s, :sidelen[3]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 3] = pixel_coords[..., 3] / (sidelen[3] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)
    pixel_coords = 2. * pixel_coords - 1.
    pixel_coords = pixel_coords.cpu().numpy().reshape(-1,3, order='F')
    return pixel_coords

def fast_random_choice(dim, num_samples_per_frame, unique=True, device='cuda:0'):
    if unique:
        num_samples = num_samples_per_frame * 2  # 防止去重后低于预定采样值
        x = torch.randint(
                0, dim[0], size=(num_samples,), device='cuda:0'
            )
        y = torch.randint(
                0, dim[1], size=(num_samples,), device='cuda:0'
            )
        z = torch.randint(
                0, dim[2], size=(num_samples,), device='cuda:0'
            )
        
        xyz = torch.stack([x, y, z], dim=-1)
        _, index = torch.unique(xyz, dim=0, sorted=False, return_inverse=True)
        xyz = xyz[index[:num_samples_per_frame, ...]]
        return xyz[...,0], xyz[...,1], xyz[...,2]
    else:
        x = torch.randint(
                0, dim[0], size=(num_samples_per_frame,), device='cuda:0'
            )
        y = torch.randint(
                0, dim[1], size=(num_samples_per_frame,), device='cuda:0'
            )
        z = torch.randint(
                0, dim[2], size=(num_samples_per_frame,), device='cuda:0'
            )
        xyz = torch.stack([x, y, z], dim=-1)
        if device == 'cpu':
            xyz = xyz.cpu()
        return xyz[...,0], xyz[...,1], xyz[...,2]
    
def count_params(model):  # 查看模型参数量
    param_num = sum(p.numel() for p in model.parameters())
    return param_num

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    
def adjust_lr(args, optimizer, epoch):
    if args.lr_s=='exp':
        lr = args.lr * math.exp(-0.02 * epoch)
    elif args.lr_s=='step':
        lr = args.lr * (0.5 ** (epoch // 50))
    elif args.lr_s == 'cosine':
        T_max = args.num_epochs
        eta_min = 0
        lr = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr