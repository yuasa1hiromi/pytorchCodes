import torch
import numpy as np
import os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


input_size = 5
output_size = 2
batch_size = 30
data_size = 3000

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        return output


rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

model = Model(input_size, output_size)
use_gpu = False
if torch.cuda.is_available():
    model = model.cuda()
    use_gpu = True
if torch.cuda.device_count() > 1:
    print("Use %d GPUs"%(torch.cuda.device_count()))
    model = nn.DataParallel(model)

for i, data in enumerate(rand_loader):
    print('epoch %d'%(i))
    if use_gpu:
        data = data.cuda()
    output = model(data)
    print("Out: input size", data.size(), "output_size", output.size())

