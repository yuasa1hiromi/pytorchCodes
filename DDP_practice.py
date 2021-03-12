import os
import os.path as osp

import torch
import torch.distributed as dist
import torch.utils.data as Data
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from utils.utils import *
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from tqdm import tqdm
import numpy as np
import logging
from datetime import datetime


########################################
# method one: use multiprocessing module
def main0():
    def example(rank, world_size):
        dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method='env://')
        model = nn.Linear(10, 10).to(rank)
        ddp_model = DDP(model, device_ids=[rank])
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

        outputs = ddp_model(torch.randn(20, 10).to(rank))
        labels = torch.randn(20, 10).to(rank)

        loss_fn(outputs, labels).backward()
        optimizer.step()
        print("over:", rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    world_size = 2
    mp.spawn(example, args=(world_size,), nprocs=world_size, join=True)

# method one: use multiprocessing module
########################################


# LeNet-5
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_dataloader(batch_size, num_workers, train, rank=-1):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.247, 0.243, 0.261)
    cifar10_pixel_mean = tuple(int(i*255) for i in cifar10_mean)
    with torch_distributed_zero_first(rank):
        if not train:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(cifar10_mean, cifar10_std)
                ])
        else:
            transform = T.Compose([
                T.RandomCrop(32, padding=4, fill=cifar10_pixel_mean),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize(cifar10_mean, cifar10_std)
            ])
        #  CIFA10 is 3 * 32 * 32
        my_dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    sampler = Data.distributed.DistributedSampler(my_dataset) if rank != -1 else None
    dataloader = Data.DataLoader(my_dataset, batch_size=batch_size, num_workers=num_workers,
                                             sampler=sampler)
    return dataloader

def evaluate_acc(model, testloader, device, loss_fn, logger):
    model.eval()
    nb_sample = len(testloader.dataset)
    s = ('%10s' + '%12s'*3) % ('Class', 'Targets', 'Loss', 'Accuracy')
    stats = []
    logger.info("Start testing ...")
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm(testloader, desc=s)):
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            loss = loss_fn(outputs, label)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.tolist()
            tcls = label.tolist()
            stats.append((predicted, tcls, [loss.item()]))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # combine each batch array to one array
    acc = np.mean(stats[0] == stats[1])  # accuracy
    mloss = np.mean(stats[2]) # mean loss
    results = {'acc': acc, 'loss': mloss}

    # print
    info = ('%10s' + '%12.6s'*3) % ('All', nb_sample, mloss, acc)
    logger.info(info)
    model.train()
    return results, s + '\n' + info



def run_train(opt, device, logger):
    epochs = opt.epochs
    rank = opt.global_rank
    train_log = opt.train_log  # training log txt path

    cuda = device.type != 'cpu'
    init_seeds(2 + rank)  # random seed is different for different DDP process
    first_rank = rank in [-1, 0]  # whether is first rank
    parallel = rank != -1   # whether use DDP mode
    model = ToyModel().to(device)

    # load weights
    load_path = None
    if first_rank and load_path is not None:
        model.load_state_dict(torch.load(load_path))   # load model before DDP

    # DDP model
    if cuda and parallel:
        model = DDP(model, device_ids=[rank], output_device=rank)

    trainloader = create_dataloader(opt.batch_size, opt.num_workers, train=True, rank=rank)
    nb = len(trainloader)
    logger.info("trainloader length: %d" % (nb))


    # set pbar, save path and testloader
    if first_rank:
        train_time_avg = MovingAverageMeter()
        test_time_avg = MovingAverageMeter()
        last_pt_path = osp.join(opt.save_dir, 'weights', 'last.pt')
        best_pt_path = osp.join(opt.save_dir, 'weights', 'best.pt')
        testloader = create_dataloader(opt.batch_size, opt.num_workers, train=False, rank=-1)
        nb_test = len(testloader)
        logger.info("testloader length: %d"%(nb_test))

    # pbar print keys
    training_show = ('%10s' + '%13s' + '%11s') % ("Epoch", 'max_gpu_mem', 'loss')

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)  # set optimizer after DDP model
    best_acc = 0

    logger.info('Start training for %d epochs...' % (epochs))
    model.train()
    for epoch in range(epochs):
        start_t = time.time()
        # randomize sampler in DDP mode
        if rank != -1:
            trainloader.sampler.set_epoch(epoch)
        # print info
        # print info for pbar
        pbar = enumerate(trainloader)   #  reset pbar to the start of trainloader for each epoch
        if first_rank:
            pbar = tqdm(pbar, total=nb)
            # reset the start point of the tracking of torch.cuda.max_memory_reserved
            torch.cuda.reset_peak_memory_stats(device)
            train_t = time_synchronized(device)
            mloss = 0
        logger.info("epoch prepare time: %.4f"%(time.time() - start_t) )
        logger.info(training_show)
        for i, (data, label) in pbar:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            # Print
            if first_rank:
                mloss = (mloss * i + loss.detach()) / (i + 1)  # update mean losses
                gpu_mem = torch.cuda.max_memory_reserved(device) / 1024**3 if torch.cuda.is_available() else 0  # (GB)
                gpu_mem_str = '%6.3gG' % gpu_mem
                s = ('%10s' + '%13s' + '%11.5g') % (
                    '%g/%g' % (epoch, epochs - 1), gpu_mem_str, mloss)
                pbar.set_description(s)

        # test on DDP process 0 or single-GPU
        if first_rank:
            train_t = time_synchronized(device) - train_t
            # acc
            test_t = time_synchronized(device)
            results, info = evaluate_acc(model, testloader, device, loss_fn, logger)
            test_t = time_synchronized(device) - test_t
            train_time_avg.update(train_t)
            test_time_avg.update(test_t)
            print_time = 'Mean train time: %.2fs.  Mean test time: %.2fs\n'%(
                    train_time_avg.avg, test_time_avg.avg)
            logger.info(print_time)
            with open(train_log, 'a') as f:
                f.write(training_show + '\n' + s + '\nTesting:\n')
                f.write(info + '\n')
                f.write(print_time + '\n')
            # save model
            if parallel:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            ckpt = {'epoch': epoch,
                    'results': results,
                    'model': state_dict,
                    }
            torch.save(ckpt, last_pt_path)
            if results['acc'] > best_acc:
                best_acc = results['acc']
                torch.save(ckpt, best_pt_path)


# DDP mode: launch with "python -m torch.distributed.launch --nproc_per_node"
def main1():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=256, help="batch size")
    parser.add_argument("-l", "--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers for dataloader")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP parameter, do not modify")
    parser.add_argument("--device", default='0', help="cuda device, i.e. 0 or 0, 1, 2, 3 or cpu")
    parser.add_argument("-s", "--save-dir", type=str, default="logs", help="model save dir")
    opt = parser.parse_args()

    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    rank = opt.local_rank   # it seems local_rank == global_rank in DDP launch mode
    first_rank = rank in [-1, 0]

    # init logger. Only rank -1 or 0 print INFO
    logging.basicConfig(format="%(message)s",
                        level=logging.INFO if first_rank else logging.WARN)
    logger = logging.getLogger()

    # set CUDA_VISIBE_DEVICE, check if batch_size is the multiple of GPU number
    # select_device function only return '0' or 'cpu'
    device = select_device(opt.device, opt.batch_size, logger)

    # DDP mode
    if rank != -1:
        assert torch.cuda.device_count() > rank, "device number should be bigger than local_rank in DDP mode"
        torch.cuda.set_device(rank)   # set device
        device = torch.device('cuda', rank)  # select device
        dist.init_process_group('nccl', init_method='env://')
        opt.batch_size = opt.total_batch_size // opt.world_size
        # To ensure get_now_time() is the same, use torch.distributed.barrier()
        torch.distributed.barrier()

    # makedir and log.txt.
    save_dir = osp.join(opt.save_dir, get_now_time())
    opt.save_dir = save_dir
    print('save dir %s  local rank:%d  global rank:%d  world size: %d'%(save_dir, rank, opt.global_rank, opt.world_size))
    opt.train_log = osp.join(save_dir, 'train_log.txt')
    logger.info("\nopt: %s\n" % str(opt))

    if first_rank:
        os.makedirs(osp.join(save_dir, 'weights'), exist_ok=True)

    run_train(opt, device, logger)

# DDP mode: launch with "python -m torch.distributed.launch --nproc_per_node"
if __name__ == "__main__":
    main1()

    pass