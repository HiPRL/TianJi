# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from drl.base.model import Model
from drl.builder import MODELS
from drl.models.utils import *

__all__ = ["MixerNet"]


@MODELS.register_module()
class MixerNet(Model):
    def __init__(
        self,
        agent_num,
        state_shape,
        mixer_dim=32,
        hypernet_dim=64,
        act: str = None,
        init_method: str = None,
        **kwargs
    ):
        super(MixerNet, self).__init__(init_method=init_method, **kwargs)
        self.agent_num = agent_num
        self.state_shape = state_shape
        self.mixer_dim = mixer_dim
        self.act = nn.ReLU() if act is None else activation_insidious_table[act.lower()]

        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_shape, hypernet_dim),
            self.act,
            nn.Linear(hypernet_dim, self.mixer_dim * self.agent_num),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_shape, hypernet_dim),
            self.act,
            nn.Linear(hypernet_dim, self.mixer_dim),
        )

        self.hyper_b1 = nn.Linear(self.state_shape, self.mixer_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_shape, self.mixer_dim),
            self.act,
            nn.Linear(self.mixer_dim, 1),
        )

    def forward(self, agents_Q, states):
        batch_size = agents_Q.size(0)
        agents_Q = agents_Q.view(-1, 1, self.agent_num)
        states = states.reshape(-1, self.state_shape)

        w1 = torch.abs(self.hyper_w1(states)).view(-1, self.agent_num, self.mixer_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.mixer_dim)

        w2 = torch.abs(self.hyper_w2(states)).view(-1, self.mixer_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        hidden = nn.functional.elu(torch.bmm(agents_Q, w1) + b1)
        q_total = (torch.bmm(hidden, w2) + b2).view(batch_size, -1, 1)
        return q_total
