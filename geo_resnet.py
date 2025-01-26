import functools
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.nn import BatchNorm1d



def get_agg_func(agg_func_type, out_channels, name = "resnet1d"):
    if agg_func_type == "mean":
        # global average pool
        return torch.mean
    elif agg_func_type == "min":
        # global min pool
        return torch.min
    elif agg_func_type == "max":
        # global max pool
        return torch.max


class ResNet1D(nn.Module):
    def __init__(self, block, num_layer_list, in_channels, out_channels, add_middle_pool=False, final_pool="mean",
                 padding_mode='zeros', dropout_rate=0.5):

        super(ResNet1D, self).__init__()

        Norm = functools.partial(BatchNorm1d)

        self.num_layer_list = num_layer_list

        self.in_channels = in_channels
        # For simplicity, make outplanes dividable by block.expansion
        assert out_channels % block.expansion == 0
        self.out_channels = out_channels
        planes = int(out_channels / block.expansion)

        self.inplanes = out_channels

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode,
                               bias=False)

        self.norm1 = Norm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        resnet_layers = []

        if len(self.num_layer_list) >= 1:
            # you have at least one number in self.num_layer_list
            layer1 = self._make_layer(block, planes, self.num_layer_list[0], Norm, padding_mode=padding_mode)
            resnet_layers.append(layer1)
            if add_middle_pool and len(self.num_layer_list) > 1:
                maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                resnet_layers.append(maxpool)

        if len(self.num_layer_list) >= 2:
            # you have at least two numbers in self.num_layer_list
            for i in range(1, len(self.num_layer_list)):
                layerk = self._make_layer(block, planes, self.num_layer_list[i], Norm, stride=2,
                                          padding_mode=padding_mode)
                resnet_layers.append(layerk)
                if add_middle_pool and i < len(self.num_layer_list) - 1:
                    maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                    resnet_layers.append(maxpool)

        self.resnet_layers = nn.Sequential(*resnet_layers)

        self.final_pool = final_pool

        self.final_pool_func = get_agg_func(agg_func_type=final_pool,
                                            out_channels=out_channels,
                                            name="resnet1d")

        self.dropout = nn.Dropout(p=dropout_rate)

        self._init_weights()

    def finalPool1d(self, x, final_pool="mean"):

        if final_pool == "mean":
            # global average pool
            # x: shape (batch_size, out_channels)
            x = self.final_pool_func(x, dim=-1, keepdim=False)
        elif final_pool == "min":
            # global min pool
            # x: shape (batch_size, out_channels)
            x, indice = self.final_pool_func(x, dim=-1, keepdim=False)
        elif final_pool == "max":
            # global max pool
            # x: shape (batch_size, out_channels)
            x, indice = self.final_pool_func(x, dim=-1, keepdim=False)
        elif final_pool.startswith("atten"):
            # attenion based aggregation
            # x: shape (batch_size, out_channels)
            x = self.final_pool_func(x)
        return x

    def forward(self, x):

        # x: shape (batch_size, out_channels, seq_len)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        # x: shape (batch_size, out_channels, seq_len/2)
        x = self.maxpool(x)

        # x: shape (batch_size, out_channels, (seq_len+k-2)/2^k )
        x = self.resnet_layers(x)

        # global pool
        # x: shape (batch_size, out_channels)
        x = self.finalPool1d(x, self.final_pool)
        x = self.dropout(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, norm, stride=1, padding_mode='circular'):

        downsample = None
        if (stride != 1) or (self.inplanes != planes * block.expansion):
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, norm, stride, downsample, padding_mode))
        # self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm, padding_mode=padding_mode))

        return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, norm, stride=1, downsample=None, padding_mode = 'circular'):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3(inplanes, planes, stride, padding_mode = padding_mode)
        self.norm1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self._conv3(planes, planes, padding_mode = padding_mode)
        self.norm2 = norm(planes)
        self.downsample = downsample
        self.stride = stride
        self.padding_mode = padding_mode

    def forward(self, x):

        residual = x

        # out: shape (batch_size, planes, (seq_len-1)/stride + 1 )
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        # out: shape (batch_size, planes, (seq_len-1)/stride + 1 )
        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            # residual: shape (batch_size, planes, (seq_len-1)/stride + 1 )
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def _conv3(self, in_planes, out_planes, stride=1, padding_mode = 'circular'):
        return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, padding_mode = padding_mode, bias=False)


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, norm, stride=1, downsample=None, padding_mode = 'circular', ):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.norm1 = norm(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, padding_mode = padding_mode, bias=False)
        self.norm2 = norm(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.norm3 = norm(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                norm(self.expansion*planes)
            )

    def forward(self, x):

        # out: shape (batch_size, planes, seq_len)
        out = F.relu(self.norm1(self.conv1(x)))
        # out: shape (batch_size, planes, (seq_len-1)/stride + 1 )
        out = F.relu(self.norm2(self.conv2(out)))
        # out: shape (batch_size, expansion*planes, (seq_len-1)/stride + 1 )
        out = self.norm3(self.conv3(out))
        # self.shortcut(x): shape (batch_size, expansion*planes, (seq_len-1)/stride + 1 )
        # out: shape (batch_size, expansion*planes, (seq_len-1)/stride + 1 )
        out += self.shortcut(x)
        out = F.relu(out)

        return out
