# -*- coding: utf-8 -*-
from utils.hook import Hook, HOOKS
from drl.utils.buffers import StepBuffer



__all__ = ['QmixPredatorHook']


@HOOKS.register_module()
class QmixPredatorHook(Hook):
    
    def before_run(self, enginer):
        module = enginer.agent.embryo
        memory_buffer = StepBuffer(module.buffer_size)
        setattr(module, 'memory', memory_buffer)

    def before_train_step(self, enginer):
        action_dim = enginer.env.action_dim
        agent_num = enginer.env.agent_num
        agents_local_obs = enginer.env.local_observations()
        return {"policy_factor": [agents_local_obs, [[1]*action_dim]*agent_num], "learn_factor": [agents_local_obs]}

    def before_val_step(self, enginer):
        action_dim = enginer.eval_env.action_dim
        agent_num = enginer.eval_env.agent_num
        agents_local_obs = enginer.eval_env.local_observations()
        return {"policy_factor": [agents_local_obs, [[1]*action_dim]*agent_num], "learn_factor": [agents_local_obs]}