# -*- coding: utf-8 -*-
from utils.hook import Hook, HOOKS
from drl.utils.buffers import EpisodeBuffer


__all__ = ['QmixHook']


@HOOKS.register_module()
class QmixHook(Hook):
    
    def before_run(self, enginer):
        module = enginer.agent.embryo
        memory_buffer = EpisodeBuffer(module.buffer_size, enginer.env.episode_limit)
        setattr(module, 'memory', memory_buffer)

    def before_train_episode(self, enginer):
        module = enginer.agent.embryo
        if not hasattr(module, 'policy_hidden_state'):
            value = module.policy_model.fc1.weight.new(1, module.policy_model.gru_dim).zero_()
            setattr(module, 'policy_hidden_state', value)
        if not hasattr(module, 'target_hidden_state'):
            value = module.target_model.fc1.weight.new(1, module.target_model.gru_dim).zero_()
            setattr(module, 'target_hidden_state', value)

    def before_train_learn(self, enginer):
        module = enginer.agent.embryo
        batch_size = enginer.agent.embryo.batch_size
        if not hasattr(module, 'batch_policy_hidden_state'):
            value = module.policy_model.fc1.weight.new(1, module.policy_model.gru_dim).zero_()
            setattr(module, 'batch_policy_hidden_state', value.unsqueeze(0).expand(batch_size, enginer.agent_num, -1))
        if not hasattr(module, 'batch_target_hidden_state'):
            value = module.target_model.fc1.weight.new(1, module.target_model.gru_dim).zero_()
            setattr(module, 'batch_target_hidden_state', value.unsqueeze(0).expand(batch_size, enginer.agent_num, -1))

    def before_train_step(self, enginer):
        if enginer.env.env._sc2_proc is not None:
            agents_actions = enginer.env.agents_avail_actions()
            agents_local_obs = enginer.env.local_observations()
            return {"policy_factor": [agents_local_obs, agents_actions], "learn_factor": [agents_actions, agents_local_obs, 1]}

    def after_train_learn(self, enginer):
        module = enginer.agent.embryo
        if hasattr(module, 'batch_policy_hidden_state'):
            del module.batch_policy_hidden_state
        if hasattr(module, 'batch_target_hidden_state'):
            del module.batch_target_hidden_state

    def after_train_episode(self, enginer):
        module = enginer.agent.embryo
        if hasattr(module, 'policy_hidden_state'):
            del module.policy_hidden_state
        if hasattr(module, 'target_hidden_state'):
            del module.target_hidden_state

    def before_val_episode(self, enginer):
        return self.before_train_episode(enginer)

    def before_val_step(self, enginer):
        if enginer.eval_env.env._sc2_proc is not None:
            agents_actions = enginer.eval_env.agents_avail_actions()
            agents_local_obs = enginer.eval_env.local_observations()
            return {"policy_factor": [agents_local_obs, agents_actions], "learn_factor": [agents_actions, agents_local_obs, 1]}

    def after_val_episode(self, enginer):
        return self.after_train_episode(enginer)