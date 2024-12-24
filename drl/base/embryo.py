# -*- coding: utf-8 -*-
import torch

from drl.base.core import EmbryoBase
from drl.base.model import Model

__all__ = ["Embryo"]


class Embryo(EmbryoBase):
    def __init__(self, model):
        super(Embryo, self).__init__()
        assert isinstance(model, Model)
        self.model = model
        self.sync_device = model.sync_device

    def set_device(self, device):
        if device is not None:
            self.sync_device = device

        if self.sync_device is None:
            return

        if isinstance(self.sync_device, str):
            self.sync_device = torch.device(self.sync_device)

        assert isinstance(self.sync_device, torch.device)

        for m in self.__dict__.values():
            if isinstance(m, Model):
                m.to_device(self.sync_device)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)
