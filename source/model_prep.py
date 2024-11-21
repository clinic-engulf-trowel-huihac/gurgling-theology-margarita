import pandas as pd
import numpy as np
import gc
import os
import time
import timm
import modelopt
import torch
import torch_tensorrt
from ptflops import get_model_complexity_info
from functools import reduce

class test_set:
    def __init__(self, args, gpu_name, results_pth, gpu_id=0, test_length=20):
        self.model_db_path = results_pth
        self.model_db = pd.read_csv(self.model_db_path)
        
        if args.type is not None:
            sets = [self.model_db['Type'].eq(typ) for typ in args.type.split(',')]
            self.test_db = self.model_db[reduce(lambda x, y: x | y, sets)]
        elif args.family is not None:
            sets = [self.model_db['Family'].eq(family) for family in args.family.split(',')]
            self.test_db = self.model_db[reduce(lambda x, y: x | y, sets)]
        elif args.model is not None:
            sets = [self.model_db['Model Name'].eq(model) for model in args.model.split(',')]
            self.test_db = self.model_db[reduce(lambda x, y: x | y, sets)]
        else:
            self.test_db = self.model_db

        if self.test_db.empty:
            raise Exception('Database does not contain any model that matches the given settings')
        self.test_db = self.test_db.reset_index(drop=True)

        # Prepare system
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.gpu_name = gpu_name
        self.gpu_id = int(gpu_id)
        torch.cuda.set_device(self.gpu_id)
        self.test_length = test_length


        if args.experiment == 'TensorRT':
            self.use_tensorrt = True
            self.data_type = torch.float16
        else:
            self.data_type = torch.float32

        self.index = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < len(self.test_db):
            # Retrive model
            self.model_name = self.test_db.loc[self.index,"Model Name"]
            self.input_size = self.test_db.loc[self.index,"Input Size"]
            self.full_name = f'{self.model_name}.{self.input_size}'

            self.max_bs = self.test_db.loc[self.index,'Max BS']
            self.done = self.test_db.loc[self.index,'DONE']

            self.index += 1

            if np.isnan(self.done):  self.done = False
            if self.done:  return

            self.model = timm.create_model(self.model_name, pretrained=True)
            self.model.to(self.device)
            self.model.eval()

        else:
            raise StopIteration
    

    def update_model_db(self, title, value):
        self.model_db = pd.read_csv(self.model_db_path)

        posn = self.model_db.loc[(self.model_db['Model Name'] == self.model_name) & (self.model_db['Input Size'] == self.input_size)].index[0]
        self.model_db.loc[posn, title] = value

        self.model_db.to_csv(self.model_db_path, index=False)

    def build_trt_engine(self, bs):
        # Check if model name contains substring
        x = torch.randn((int(bs), 3, self.input_size, self.input_size), device=self.device, dtype=self.data_type)
        if 'hgnet' in self.model_name:
            compile_settings = {
                'inputs' : [x],
                'enabled_precisions': {self.data_type},
                'device': torch_tensorrt.Device(
                    device_type=torch_tensorrt.DeviceType.GPU,
                    gpu_id=self.gpu_id,
                ),
                'debug': False,
                'sparse_weights' : False,
            }
            torch._dynamo.reset()
            self.model = torch.jit.script(self.model)
            self.model = torch_tensorrt.compile(self.model, **compile_settings)
            self.model(x)
        else:
            compile_settings = {
                'inputs' : [x],
                'enabled_precisions': {self.data_type},
                'device': torch_tensorrt.Device(
                    device_type=torch_tensorrt.DeviceType.GPU,
                    gpu_id=self.gpu_id,
                ),
                'debug': False,
                'disable_tf32' : True,
                'strict_types' : True,
                'use_cuda_graph' : True,
                'sparse_weights' : False,
                'optimization_level' : 3,
                'hardware_compatible' : False,
                'pass_through_build_failures' : True,
                'ir' : "dynamo",
            }
            torch._dynamo.reset()
            self.model = torch.compile(self.model, backend='torch_tensorrt', options=compile_settings, dynamic=False)
            self.model(x)

        # delete x
        x.to('cpu')
        del x
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def infer(self, bs):
        timestamps = [time.time_ns()]
        while(timestamps[-1] - timestamps[0] < self.test_length*1e9 or len(timestamps) < 13):
            self.model(torch.randn(int(bs), 3, self.input_size, self.input_size, device=self.device, dtype=self.data_type))
            torch.cuda.synchronize()
            timestamps.append(time.time_ns())
        
        return timestamps
    
    def forward_pass(self, bs):
        self.model(torch.randn(int(bs), 3, self.input_size, self.input_size, device=self.device))
        torch.cuda.synchronize()

    def summary(self):
        print(self.model)

    def reload(self):
        self.model.to('cpu')
        del self.model
        self.model = None
        gc.collect()
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        os.system('nvidia-smi')
        
        self.model = timm.create_model(self.model_name, pretrained=True)

        self.model = self.model.eval().to(self.device).to(dtype=self.data_type) 
        torch.cuda.synchronize()

    def clean_up(self):
        self.model.to('cpu')
        del self.model
        self.model = None
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        os.system('nvidia-smi')

    def reset(self):
        self.index = 0

class bs_gen:
    def __init__(self, mode, grain):
        self.mode = mode
        self.grain = grain
        self.i = 0
        self.size = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.mode == 'lin':
            if self.size <= self.grain:
                self.size = 2**self.i
            else:
                self.size += self.grain
        elif self.mode == 'exp':
            if self.i < self.grain:
                self.size += 1
            else:
                self.size += 2**(int(self.i/self.grain)-1)
        self.i += 1
    
    def __str__(self):
        return str(self.size)

    def __int__(self):
        return self.size

    def __add__(self, x):
        return self.size + x

    def __rsub__(self, x):
        return x - self.size

    def __mul__(self, x):
        return self.size * x

    def __rtruediv__(self, x):
        return x / self.size

    def __itruediv__(self, x):
        return self.size / x

    def step_back(self):
        if self.mode == 'lin':
            return self.size - self.grain
        elif self.mode == 'exp':
            return int(self.size - 2**(int((self.i-1)/self.grain)-1))

    def reset(self, grain = None):
        self.i = 0
        self.size = 0
        if grain is not None:
            self.grain = grain
