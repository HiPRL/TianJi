# -*- coding: utf-8 -*-
import numpy as np
from utils.common import function

from drl.builder import AGENTS
from drl.base.agent import Agent



__all__ = ['PPO']



@AGENTS.register_module()
class PPO(Agent):
    def __init__(self, embryo, alg_type='on-policy'):
        super(PPO, self).__init__(embryo, alg_type)

    def predict(self, status, _=None):
        action, probs = self.embryo.execute(np.array(status))
        return action.cpu().detach().numpy().squeeze(), probs.cpu().detach().numpy().squeeze()
    
    def learn(self, *args, **kwargs):
        if args:
            self.embryo.save_memory(*args)
            self.alg_scalar_data['train_memory_size'] = len(self.embryo.memory)
        
        if self.is_learn():
            loss = self.embryo.update()
            self.learn_step += 1
            self.alg_scalar_data['train_step_loss'] = loss

            return loss, self.learn_step
    
    def is_learn(self):
        return self.embryo.memory.is_overflow if self.embryo.warmup_size is None else len(self.embryo.memory) > self.embryo.warmup_size

    def policy(self, status, action_dim, explore_step=None):
        if self.embryo.warmup_full_random and not self.is_learn() and explore_step is None:
            return function(np.random.randint, action_dim)(), None
        else:
            return self.predict(status)
    
    def save(self, save_path, use_full_path=False):
        save_path = str(save_path) if use_full_path else str(save_path / (self.__class__.__name__ + '.pt'))
        super().save(save_path, self.embryo.policy_model)