import os
import os.path as osp

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from utils.utils import select_device, time_synchronized, torch_distributed_zero_first, init_seeds,\
    MovingAverageMeter
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from tqdm import tqdm
import numpy as np
import logging
from datetime import datetime


########################################
# method one: use multiprocessing module
def example(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method='env://')
    model = nn.Linear(10, 10).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr = 0.001)

    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)

    loss_fn(outputs, labels).backward()
    optimizer.step()
    print("over:", rank)


def main0():
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


def create_dataloader(batch_size, num_workers, rank, train):
    with torch_distributed_zero_first(rank):
        if not train:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        else:
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        my_trainset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(my_trainset) if rank != -1 else None
    dataloader = torch.utils.data.DataLoader(my_trainset, batch_size=batch_size, num_workers=num_workers,
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
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)  # random seed is different for different DDP process
    is_first_rank = rank in [-1, 0]  # whether is first rank
    is_parallel = rank != -1   # whether use DDP mode
    model = ToyModel().to(device)

    # load weights
    load_path = None
    if is_first_rank and load_path is not None:
        model.load_state_dict(torch.load(load_path))   # load model before DDP

    # DDP model
    if cuda and is_parallel:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    trainloader = create_dataloader(opt.batch_size, opt.num_workers, rank, train=True)

    # save training log txt
    if is_first_rank:
        train_time_avg = MovingAverageMeter()
        test_time_avg = MovingAverageMeter()
        training_log_txt = open(opt.save_txt, 'w')  # training log txt path
        last_pt_path = osp.join(opt.save_dir, 'weights', 'last.pt')
        best_pt_path = osp.join(opt.save_dir, 'weights', 'best.pt')
        testloader = create_dataloader(opt.batch_size, opt.num_workers, rank, train=False)
        nb_test = len(testloader)
        logger.info("testloader length: %d"%(nb_test))

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr) # set optimizer after DDP model
    nb = len(trainloader)
    best_acc = 0

    # print
    training_show = ('%10s' + '%13s' + '%11s') % ("Epoch", 'max_gpu_mem', 'loss')
    logger.info('Start training for %d epochs...' % (epochs))
    model.train()

    for epoch in range(epochs):

        if rank != -1:
            trainloader.sampler.set_epoch(epoch)
        pbar = enumerate(trainloader)
        logger.info(training_show)

        # print info
        if is_first_rank:
            train_t = time_synchronized(device)
            pbar = tqdm(pbar, total=nb)
            max_gpu_mem = 0   # max gpu memory
            mloss = 0

        for i, (data, label) in pbar:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            # Print
            if is_first_rank:
                mloss = (mloss * i + loss.detach()) / (i + 1)  # update mean losses
                gpu_mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
                max_gpu_mem = max(max_gpu_mem, gpu_mem)
                max_gpu_mem_str = '%6.3gG' % max_gpu_mem
                s = ('%10s' + '%13s' + '%11.5g') % (
                    '%g/%g' % (epoch, epochs - 1), max_gpu_mem_str, mloss)
                pbar.set_description(s)

        # test on DDP process 0 or single-GPU
        if is_first_rank:
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
            training_log_txt.write(training_show + '\n' + s + '\nTesting:\n')
            training_log_txt.write(info + '\n')
            training_log_txt.write(print_time + '\n')
            training_log_txt.flush()
            # save model
            if is_parallel:
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

    if is_first_rank:
        training_log_txt.close()


def main1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='total train epochs')
    parser.add_argument('--num-workers', type=int, default=4, help='')
    parser.add_argument('--local-rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', default='logs', help='model save dir')
    opt = parser.parse_args()

    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    # init logger. Only rank -1 or 0 print INFO
    logging.basicConfig(format="%(message)s",
                        level=logging.INFO if opt.local_rank in [-1, 0] else logging.WARN)
    logger = logging.getLogger()
    logger.info("\nopt: %s\n"%str(opt))
    rank = opt.local_rank

    # make save dirs
    if rank in [-1, 0]:
        now_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        opt.save_dir = osp.join(opt.save_dir, now_time)
        opt.save_txt = osp.join(opt.save_dir, 'training_log.txt')
        os.makedirs(osp.join(opt.save_dir, 'weights'), exist_ok=True)

    if rank == -1:
        device = select_device(opt.device, batch_size=opt.batch_size) # batch size should be multiple of GPU number

    else:  # DDP mode
        assert torch.cuda.device_count() > opt.local_rank, 'device number should be bigger than local_rank in DDP mode'
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    run_train(opt, device, logger)




if __name__ == "__main__":
    main1()