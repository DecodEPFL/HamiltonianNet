import torch
import math
from torch.utils import data


# 2D data-set generators

def _data_shuffle(data2d, label):
    data_size = data2d.shape[0]
    randindex = torch.randperm(data_size)
    data2d = data2d[randindex, :, :]
    label = label[randindex, :]
    return data2d, label


def _data_extension(data2d, nf, input_ch=None):
    if nf < 2:
        print("Dimension not valid")
        return
    elif nf % 2 == 1:
        print("Using odd dimension nf")
    data_size = data2d.shape[0]
    if input_ch is not None:
        # input_ch is a list of two elements. The elements indicate where the data enters.
        idx_x = input_ch[0]
        idx_y = input_ch[1]
    else:
        idx_x = 0
        idx_y = nf-1
    data2d = torch.cat((torch.zeros(data_size, idx_x-0, 1),
                        data2d[:, 0:1, :],
                        torch.zeros(data_size, idx_y-idx_x-1, 1),
                        data2d[:, 1:2, :],
                        torch.zeros(data_size, nf-1-idx_y, 1)), 1)
    return data2d


def swiss_roll(data_size, shuffle=True, nf=2, noise_std=0, input_ch=None):

    data2d = torch.zeros(data_size, 2, 1)
    label = torch.ones(data_size, 1)
    label[math.floor(data_size / 2):, :] = 0

    r1 = torch.linspace(0, 1, math.ceil(data_size / 2))
    r2 = torch.linspace(0.2, 1.2, math.ceil(data_size / 2))
    theta = torch.linspace(0, 4 * math.pi - 4 * math.pi / math.ceil(data_size / 2), math.ceil(data_size / 2))
    data2d[0:math.ceil(data_size / 2), 0, 0] = r1 * torch.cos(theta)
    data2d[0:math.ceil(data_size / 2), 1, 0] = r1 * torch.sin(theta)
    data2d[math.floor(data_size / 2):, 0, 0] = r2 * torch.cos(theta)
    data2d[math.floor(data_size / 2):, 1, 0] = r2 * torch.sin(theta)
    if noise_std:
        for i in range(2):
            data2d[:, i, 0] = data2d[:, i, 0] + noise_std*torch.randn(data_size)

    if shuffle:
        data2d, label = _data_shuffle(data2d, label)
    
    if nf != 2:
        data2d = _data_extension(data2d, nf, input_ch)
    
    domain = [-1.2, 1.2, -1.2, 1.2]
    return data2d, label, domain


def double_circles(data_size, shuffle=True, nf=2, noise_std=0, input_ch=None):

    data2d = torch.zeros(data_size, 2, 1)
    label = torch.zeros(data_size, 1)

    for i in range(int(data_size / 4)):
        theta = torch.tensor(i / int(data_size / 4) * 4 * 3.14)

        r = 1
        label[i, :] = 0
        data2d[i, :, :] = torch.tensor(
            [[r * torch.cos(theta) + 0.6 * (torch.rand(1) - 0.5)],
             [r * torch.sin(theta) + 0.6 * (torch.rand(1) - 0.5)]])

        r = 2
        label[i + int(data_size / 4), :] = 1
        data2d[i + int(data_size / 4), :, :] = torch.tensor(
            [[r * torch.cos(theta) + 0.6 * (torch.rand(1) - 0.5)],
             [r * torch.sin(theta) + 0.6 * (torch.rand(1) - 0.5)]])

        r = 3
        label[i + int(2 * data_size / 4), :] = 0
        data2d[i + int(2 * data_size / 4), :, :] = torch.tensor(
            [[r * torch.cos(theta) + 0.6 * (torch.rand(1) - 0.5)],
             [r * torch.sin(theta) + 0.6 * (torch.rand(1) - 0.5)]])

        r = 4
        label[i + int(3 * data_size / 4), :] = 1
        data2d[i + int(3 * data_size / 4), :, :] = torch.tensor(
            [[r * torch.cos(theta) + 0.6 * (torch.rand(1) - 0.5)],
             [r * torch.sin(theta) + 0.6 * (torch.rand(1) - 0.5)]])

    if noise_std:
        for i in range(2):
            data2d[:, i, 0] = data2d[:, i, 0] + noise_std*torch.randn(data_size)
    
    if shuffle:
        data2d, label = _data_shuffle(data2d, label)

    if nf != 2:
        data2d = _data_extension(data2d, nf, input_ch)
    
    domain = [-5, 5, -5, 5]
    return data2d, label, domain


def double_moons(data_size, shuffle=True, nf=2, noise_std=0, input_ch=None):

    data2d = torch.zeros(data_size, 2, 1)
    label = torch.zeros(data_size, 1)

    for i in range(int(data_size / 4)):
        theta = torch.tensor(i / int(data_size / 4) * 3.14)

        label[i, :] = 0
        data2d[i, :, :] = torch.tensor(
            [[torch.cos(theta) + 0.3*(torch.rand(1)-0.5)], [torch.sin(theta) + 0.3*(torch.rand(1)-0.5)]])

        label[i+int(data_size / 4), :] = 1
        data2d[i+int(data_size / 4), :, :] = torch.tensor(
            [[torch.ones(1) - torch.cos(theta) + 0.3*(torch.rand(1)-0.5)],
             [torch.ones(1)*0.5 - torch.sin(theta) + 0.3*(torch.rand(1)-0.5)]])

        label[i + int(data_size / 2), :] = 0
        data2d[i + int(data_size / 2), :, :] = torch.tensor(
            [[torch.cos(theta) + 0.3 * (torch.rand(1) - 0.5) + 2*torch.ones(1)],
             [torch.sin(theta) + 0.3 * (torch.rand(1) - 0.5)]])

        label[i + int(data_size * 3 / 4), :] = 1
        data2d[i + int(data_size * 3 / 4), :, :] = torch.tensor(
            [[torch.ones(1) - torch.cos(theta) + 0.3 * (torch.rand(1) - 0.5) + 2*torch.ones(1)],
             [torch.ones(1) * 0.5 - torch.sin(theta) + 0.3 * (torch.rand(1) - 0.5)]])
    
    if noise_std:
        for i in range(2):
            data2d[:, i, 0] = data2d[:, i, 0] + noise_std*torch.randn(data_size)
    
    if shuffle:
        data2d, label = _data_shuffle(data2d, label)

    if nf != 2:
        data2d = _data_extension(data2d, nf, input_ch)
    
    domain = [-2, 5, -1, 2]
    return data2d, label, domain


class Dataset(data.Dataset):

    def __len__(self):
        return len(self.list_ids)

    def __init__(self, list_ids, data_in, labels):
        self.list_ids = list_ids
        self.data = data_in
        self.labels = labels

    def __getitem__(self, index):

        idx = self.list_ids[index]

        x = self.data[idx, :, :]
        y = self.labels[idx, :]

        return x, y
