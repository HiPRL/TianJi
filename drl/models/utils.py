# -*- coding: utf-8 -*-
import torch.nn as nn



activation_insidious_table = {
    "relu": nn.ReLU(),
    "relu6": nn.ReLU6(),
    "lrelu": nn.LeakyReLU(),
    "prelu": nn.PReLU(),
    "elu": nn.ELU(),
    #"silu": nn.SiLU(),
    #"gelu": nn.GELU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
}


class Residual(nn.Module):
    def __init__(self, fn):
        self.fn = fn
        super().__init__()
    
    def forward(self, x):
        return self.fn(x) + x


def constant_init(module, val=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


_init_registry_table = [kaiming_init, uniform_init, normal_init, xavier_init, constant_init]

init_table = {}
for item in _init_registry_table:
    init_table[item.__name__] = item