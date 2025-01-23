# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from drl.models.utils import *
from drl.builder import MODELS
from drl.base.model import Model



__all__ = ['PPOAtari']


@MODELS.register_module()
class PPOAtari(Model):
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_actions = action_dim
        
        network = [
            torch.nn.Conv2d(state_dim, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim + 1)
        ]
        
        self.network = nn.Sequential(*network)
        self.softmax = nn.Softmax(dim=1)

    def init_params(self):
        init_func = kaiming_init
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                init_func(m)

    def forward(self, states):
        policy, value = torch.split(self.network(states),(self.num_actions, 1), dim=1)
        policy = self.softmax(policy)
        return policy, value
    

