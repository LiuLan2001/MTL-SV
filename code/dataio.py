import torch.utils
import numpy as np
import torch
import os
import threading
import queue
import tqdm
from concurrent.futures import ThreadPoolExecutor
from utils import BatchIndex,get_mgrid,fast_random_choice,count_params,cleanup,seed_everything,adjust_lr

class ScalarDataSet():
    def __init__(self,args, device='cuda:0'):
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.interval = args.interval
        self.downsample_factor = args.downsample_factor
        self.device = device

        if self.dataset == 'tornado':
            self.dim = [128, 128, 128]
            self.total_samples = 1
            self.data_path = './dataset/tornado/'        
        elif self.dataset == 'fivejets':
            self.dim = [128, 128, 128]
            self.total_samples = 1
            self.data_path = './dataset/fivejets/' 
        elif self.dataset == 'halfcy':
            self.dim = [640, 240, 80]
            self.total_samples = 1
            self.data_path = './dataset/halfcy/'  
        elif self.dataset == 'h2':
            self.dim = [600, 248, 248]
            self.total_samples = 1
            self.data_path = './dataset/h2/' 
        elif self.dataset == 'combustion':
            self.dim = [480, 720, 120]  
            self.total_samples = 1
            self.data_path = './dataset/combustion/'

        self.num_workers = 16

        self.samples = [i for i in range(1,self.total_samples+1,self.interval+1)]
        self.total_samples = self.samples[-1]
        self.num_samples_per_frame = (self.dim[0]*self.dim[1]*self.dim[2]//self.downsample_factor)//self.batch_size * self.batch_size

        self.queue_size = 2
        self.loader_queue = queue.Queue(maxsize=self.queue_size)  
        self.executor = ThreadPoolExecutor(max_workers=self.queue_size)

        if args.mode == 'train':
            self.data = self.preload_with_multi_threads(self.load_volume_data, num_workers=self.num_workers, data_str='Volume Data')
            self.data = torch.as_tensor(np.asarray(self.data), device=self.device)  

            self.len = self.num_samples_per_frame * len(self.samples)
            self._get_data = self._get_training_data

            samples = self.dim[2]*self.dim[1]*self.dim[0]
            self.coords = get_mgrid([self.dim[0],self.dim[1],self.dim[2]],dim=3)
            self.time = np.zeros((samples,1))
            self.testing_data_inputs = torch.as_tensor(np.concatenate((self.time, self.coords),axis=1), dtype=torch.float, device='cuda:0')
            self.preload_data()

        elif args.mode == 'inf':
            samples = self.dim[2]*self.dim[1]*self.dim[0]
            self.coords = get_mgrid([self.dim[0],self.dim[1],self.dim[2]],dim=3)
            self.time = np.zeros((samples,1))
            self.testing_data_inputs = torch.as_tensor(np.concatenate((self.time, self.coords),axis=1), dtype=torch.float, device='cuda:0')
            
    def preload_data(self):
        if self.loader_queue.full():
            return  
        self.loader_queue.put(self._get_data())

    def get_data(self):
        if self.loader_queue.empty():
            print("DataLoader is not ready yet! Waiting...")
        while self.loader_queue.empty():
            pass
        current_data = self.loader_queue.get()
        self.executor.submit(self.preload_data)
        return current_data

    @torch.no_grad()
    def _get_testing_data(self, idx):
        t = idx - 1
        t = t / max((self.total_samples-1), 1)
        t = 2.0 * t - 1.0
        testing_data_inputs = self.testing_data_inputs.clone()
        testing_data_inputs[:,0] = t
        batchidxgenerator = BatchIndex(testing_data_inputs.shape[0], self.batch_size, False)
        return testing_data_inputs, batchidxgenerator

    @torch.no_grad()
    def _get_training_data(self):
        training_data_inputs = []
        training_data_outputs = []

        for i in range(0, len(self.samples)):
            x,y,z = fast_random_choice(self.dim, self.num_samples_per_frame)
            t = torch.ones_like(x) * (self.samples[i]-1)

            outputs = self.data[i, x, y, z]  
            x = x / (self.dim[0] - 1)
            y = y / (self.dim[1] - 1)
            z = z / (self.dim[2] - 1)
            t = t / max((self.total_samples-1), 1)

            inputs = torch.stack([t, x, y, z], dim=-1)
            inputs = 2.0 * inputs - 1.0  
            training_data_inputs.append(inputs)
            training_data_outputs.append(outputs)

        training_data_inputs = torch.cat(training_data_inputs, dim=0).cuda()
        training_data_outputs = torch.cat(training_data_outputs, dim=0).cuda()
        idx = torch.randperm(training_data_inputs.shape[0], device='cpu')
        training_data_inputs = training_data_inputs[idx].contiguous()
        training_data_outputs = training_data_outputs[idx].contiguous()
        batchidxgenerator = BatchIndex(self.len, self.batch_size, shuffle=True)
        del idx
        cleanup()
        return training_data_inputs, training_data_outputs, batchidxgenerator

    def load_volume_data(self, idx):
        d = np.fromfile(self.data_path+'{:04d}.raw'.format(self.samples[idx]), dtype='<f')
        d = 2. * (d - np.min(d)) / (np.max(d) - np.min(d)) - 1.  
        d = d.reshape(self.dim[2],self.dim[1],self.dim[0])  
        d = d.transpose(2,1,0)  
        return d

    def _preload_worker(self, data_list, load_func, q, lock, idx_tqdm):
        # Keep preloading data in parallel.
        while True:
            idx = q.get()
            data_list[idx] = load_func(idx)
            with lock:
                idx_tqdm.update()
            q.task_done()

    def preload_with_multi_threads(self, load_func, num_workers, data_str='images'):
        data_list = [None] * len(self.samples)

        q = queue.Queue(maxsize=len(self.samples))
        idx_tqdm = tqdm.tqdm(range(len(self.samples)), desc=f"Loading {data_str}", leave=False)
        for i in range(len(self.samples)):
            q.put(i)
        lock = threading.Lock()
        for ti in range(num_workers):
            t = threading.Thread(target=self._preload_worker,
                                    args=(data_list, load_func, q, lock, idx_tqdm), daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        assert all(map(lambda x: x is not None, data_list))

        return data_list  