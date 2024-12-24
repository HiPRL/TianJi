# -*- coding: utf-8 -*-
from typing import Any

import torch
import torch.nn as nn

from drl.base.core import ModelBase

__all__ = ["Model"]


class Model(nn.Module, ModelBase):
    def __init__(self, *args, device: Any = None, init_method: str = None, **kwargs):
        self.sync_device = device
        self.init_method = init_method
        super(Model, self).__init__()

    def to_device(self, device=None):
        if device is not None:
            self.sync_device = device

        if self.sync_device is None:
            return

        if isinstance(self.sync_device, str):
            self.sync_device = torch.device(self.sync_device)
            self.to(self.sync_device)
        elif isinstance(self.sync_device, torch.device):
            self.to(self.sync_device)
        else:
            raise TypeError("unknown device type")

    def init_params(self, *args, **kwargs):
        from drl.models.utils import init_table

        init_func = None
        if self.init_method is not None and self.init_method in init_table.keys():
            init_func = init_table[self.init_method]

        if init_func:
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    init_func(m)

    def sync_weights_to(self, target_model, weight=1.0):
        assert isinstance(
            target_model, Model
        ), f"target_model type must be 'Model', but got {type(target_model)}"
        assert (
            self.__class__.__name__ == target_model.__class__.__name__
        ), "must be the same class for params sync warp"
        assert not target_model is self, "cannot copy between identical model"
        assert weight >= 0 and weight <= 1, "weight between in range of 0 ~ 1"

        target_vars = dict(target_model.named_parameters())
        for name, var in self.named_parameters():
            target_vars[name].data.copy_(
                weight * var.data + (1 - weight) * target_vars[name].data
            )

    def get_weights(self):
        weights = self.state_dict()
        for key in weights.keys():
            weights[key] = weights[key].cpu().numpy()
        return weights

    def set_weights(self, weights):
        new_weights = dict()
        for key in weights.keys():
            new_weights[key] = torch.from_numpy(weights[key]).to(self.sync_device)
        self.load_state_dict(new_weights)
