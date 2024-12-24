# -*- coding: utf-8 -*-
import torch
import numpy as np
from copy import deepcopy
from utils.common import function

from drl.builder import AGENTS
from drl.base.agent import Agent



__all__ = ['MADDPG']



@AGENTS.register_module()
class MADDPG(Agent):
    def __init__(self, embryo, agent_num):
        super(MADDPG, self).__init__(embryo)
        self.loss = 0
        self.agent_num = agent_num
        self.agents = [deepcopy(self.embryo) for i in range(agent_num)]

    def predict(self, status, use_target_model=False):
        pred = []
        status = status.squeeze()
        for i, agent in enumerate(self.agents):
            pred.append(agent.execute(status[i], use_target_model).cpu().detach().numpy().squeeze())
        pred = np.vstack(tuple(map(lambda item: np.asarray(item), pred)))
        return pred.reshape((1,) + pred.shape)
    
    def learn(self, *args, **kwargs):
        if args:
            self.memory.push(*args)
            self.alg_scalar_data['train_memory_size'] = len(self.memory)

        if self.is_learn():
            if self.learn_step % self.embryo.update_target_iter == 0:
                for agent in self.agents:
                    agent.sync_update_weights()

            self.multi_update()
            self.learn_step += 1
            self.alg_scalar_data['train_step_loss'] = self.loss

            return self.loss, self.learn_step

    def is_learn(self):
        return self.memory.is_overflow if self.embryo.warmup_size is None else len(self.memory) > self.embryo.warmup_size

    def policy(self, status, explore_step=None):
        return self.embryo.explorer.sample(function(self.predict, status), to_numpy=True)
    
    def multi_update(self):
        self.loss = 0
        batch_state, batch_action, batch_reward, batch_next_state, batch_terminal, _ = self.memory.sample_batch(self.embryo.batch_size)
        batch_next_action = torch.zeros(batch_action.shape)
        for agent_index, agent in enumerate(self.agents):
            batch_next_action[:,agent_index,:] = agent.execute(batch_next_state[:,agent_index,:], use_target_model=True).cpu().detach()
        
        for agent_index, agent in enumerate(self.agents):
            self.loss += agent.update(batch_state, batch_action, batch_reward[:,agent_index,:], batch_next_state, batch_next_action, batch_terminal[:,agent_index,:], agent_index)

    def update_model_params(self, weights):
        [agent.policy_model.set_weights(weights[i]) for i, agent in enumerate(self.agents)]
    
    def take_model_params(self):
        return [agent.policy_model.get_weights() for agent in self.agents]

    def save(self, save_path, use_full_path=False):
        save_path = str(save_path) if use_full_path else str(save_path / (self.__class__.__name__ + '.pt'))
        model_dict = {'agent_{}'.format(agent_index):agent.policy_model for agent_index, agent in enumerate(self.agents)}
        super().save(save_path, model_dict)
    
    def resume(self, save_path):
        model_dict = torch.load(save_path)
        for agent_index, agent in enumerate(self.agents):
            agent.policy_model.load_state_dict(model_dict['agent_{}'.format(agent_index)].state_dict())