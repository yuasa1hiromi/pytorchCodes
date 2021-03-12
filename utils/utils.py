import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import logging
import os
from contextlib import contextmanager
import random
import numpy as np
import math
import datetime

def get_now_time(fmt='%Y_%m_%d-%H_%M_%S'):
    return datetime.datetime.now().strftime(fmt)

def time_synchronized(device=None):
    torch.cuda.synchronize(device) if torch.cuda.is_available() else None
    return time.time()

class MovingAverageMeter:
    """Computes and stores moving average and current value"""
    def __init__(self, range = 200):
        assert range > 1, "moving range can't be less than 1"
        self.dr = 1 - 1 / range  # decay rate
        self.reset()

    def reset(self):
        self.val = self.avg0 =  self.count = self.avg = 0

    def update(self, val):
        self.val = val
        self.count += 1
        dr = self.dr
        self.avg0 = (dr * self.avg0 + (1-dr)*val)
        self.avg = self.avg0 / (1 - math.pow(dr, self.count))

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

def my_print_logger(info, logger=None):
    if logger:
        logger.info(info)
    else:
        print(info)

# set CUDA_VISIBLE_DEVICES and check batch size
def select_device(device='', batch_size=None, logger=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print_info = "%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c)
            my_print_logger(print_info, logger)
    else:
        my_print_logger(f'Using torch {torch.__version__} CPU', logger)
    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)

if __name__ == "__main__":
    re = int(os.environ['RANK'])
    print(re)


    pass