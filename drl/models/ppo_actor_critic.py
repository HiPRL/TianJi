# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from drl.models.utils import *
from drl.builder import MODELS
from drl.base.model import Model



__all__ = ['PPOActorCritic']



@MODELS.register_module()
class PPOActorCritic(Model):
    def __init__(self, state_dim, action_dim, mlp_dim, init_method: str = None, **kwargs):
        super(PPOActorCritic, self).__init__(init_method=init_method, **kwargs)
        self.num_actions = action_dim

        network = [
            torch.nn.Linear(state_dim, mlp_dim),
            nn.ReLU(),
            torch.nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            torch.nn.Linear(mlp_dim, action_dim + 1),
        ]

        self.network = nn.Sequential(*network)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, states):
        policy, value = torch.split(self.network(states),(self.num_actions, 1), dim=1)
        policy = self.softmax(policy)
        return policy, value