# -*- coding: utf-8 -*-
from utils.hook import Hook, HOOKS
from drl.utils.buffers import StepBuffer


__all__ = ['MADDPGHook']


@HOOKS.register_module()
class MADDPGHook(Hook):
    
    def before_run(self, enginer):
        memory_buffer = StepBuffer(enginer.agent.embryo.buffer_size)
        setattr(enginer.agent, 'memory', memory_buffer)

    def before_train_step(self, enginer):
        return {"policy_factor": [enginer.status], "learn_factor": [None]}

    def before_val_step(self, enginer):
        return {"policy_factor": [enginer.eval_status], "learn_factor": [None]}