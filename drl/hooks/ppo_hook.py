# -*- coding: utf-8 -*-
from utils.hook import Hook, HOOKS


__all__ = ['PPOHook']


@HOOKS.register_module()
class PPOHook(Hook):
    def __init__(self):
        pass
    
    def before_train_step(self, enginer):
        return {"policy_factor": [enginer.status, enginer.env.action_dim], "learn_factor": [None]}

    def before_val_step(self, enginer):
        return {"policy_factor": [enginer.eval_status, enginer.eval_env.action_dim], "learn_factor": [None]}