# -*- coding: utf-8 -*-
import torch.nn as nn
from drl.base.model import Model
from drl.builder import MODELS
from drl.models.utils import *

__all__ = ["GRU"]


@MODELS.register_module()
class GRU(Model):
    def __init__(
        self, c1, c2, gru_dim=64, act: str = None, init_method: str = None, **kwargs
    ):
        super(GRU, self).__init__(init_method=init_method, **kwargs)
        self.gru_dim = gru_dim
        self.fc1 = nn.Linear(c1, gru_dim)
        self.gru_nn = nn.GRUCell(gru_dim, gru_dim)
        self.fc2 = nn.Linear(gru_dim, c2)
        self.act = nn.ReLU() if act is None else activation_insidious_table[act.lower()]

    def forward(self, x, hidden_state):
        x = self.act(self.fc1(x))
        h = self.gru_nn(
            x.reshape(-1, self.gru_dim), hidden_state.reshape(-1, self.gru_dim)
        )
        q = self.fc2(h)
        return q, h
