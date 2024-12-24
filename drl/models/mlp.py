# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from drl.base.model import Model
from drl.builder import MODELS
from drl.models.utils import *

__all__ = ["MLP"]


@MODELS.register_module()
class MLP(Model):
    def __init__(
        self,
        c1,
        c2,
        mlp_dim=(256,),
        act: str = None,
        dueling: bool = False,
        init_method: str = "xavier_init",
        **kwargs
    ):
        super(MLP, self).__init__(init_method=init_method, **kwargs)
        self.dueling = dueling
        self.act = nn.GELU() if act is None else activation_insidious_table[act.lower()]

        mlp_dim = [mlp_dim] if isinstance(mlp_dim, int) else mlp_dim
        assert isinstance(mlp_dim, (tuple, list))
        self.fc1 = nn.Linear(c1, mlp_dim[0])
        _h = []
        for num in mlp_dim:
            _h.append(nn.Linear(num, num))
            _h.append(self.act)
        self.hidden_layers = nn.Sequential(*_h)
        self.fc2 = nn.Linear(mlp_dim[-1], mlp_dim[-1])
        self.out = nn.Linear(mlp_dim[-1], c2)
        if self.dueling:
            self.duel_fc2 = nn.Linear(mlp_dim[-1], mlp_dim[-1])
            self.duel_out = nn.Linear(mlp_dim[-1], 1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.hidden_layers(x)
        x = self.act(self.fc2(x))
        y = self.out(x)
        if self.dueling:
            x = self.act(self.duel_fc2(x))
            v = self.duel_out(x)
            y = v + (y - torch.mean(y, dim=1, keepdim=True))
        return y
