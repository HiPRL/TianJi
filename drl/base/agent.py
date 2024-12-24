# -*- coding: utf-8 -*-
import os

import torch

from drl.base.core import AgentBase
from drl.base.embryo import Embryo

__all__ = ["Agent"]


class Agent(AgentBase):
    def __init__(self, embryo, alg_type="off-policy"):
        super(Agent, self).__init__()
        assert isinstance(embryo, Embryo)
        self.embryo = embryo
        self.learn_step = 0
        self.alg_type = alg_type
        self.alg_scalar_data = {}

    def is_learn(self, *args, **kwargs):
        raise NotImplementedError()

    def set_device(self, device):
        self.embryo.set_device(device)
        torch.cuda.empty_cache()

    def save(self, save_path, model=None):
        if model is None:
            model = self.embryo.model
        dirname = os.sep.join(save_path.split(os.sep)[:-1])
        if dirname != "" and not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save(model, save_path)

    def resume(self, save_path, model=None, map_location=None):
        if model is None:
            model = self.embryo.model
        checkpoint = torch.load(save_path, map_location=map_location)
        model.load_state_dict(
            checkpoint
            if not hasattr(checkpoint, "state_dict")
            else checkpoint.state_dict()
        )

    def update_model_params(self, weights):
        self.embryo.set_weights(weights)

    def take_model_params(self):
        return self.embryo.get_weights()
