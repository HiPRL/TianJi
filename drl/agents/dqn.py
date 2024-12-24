# -*- coding: utf-8 -*-
import numpy as np
from utils.common import function

from drl.builder import AGENTS
from drl.base.agent import Agent



__all__ = ['DQN']



@AGENTS.register_module()
class DQN(Agent):
    def __init__(self, embryo):
        super(DQN, self).__init__(embryo)

    def predict(self, status, action_dim):
        pred = self.embryo.execute(np.array(status)).cpu().detach().numpy().squeeze()
        best_actions = np.where(pred == pred.max())[0]
        action = np.random.choice(best_actions)
        return action
    
    def learn(self, *args, **kwargs):
        if args:
            self.embryo.save_memory(*args)
            self.alg_scalar_data['train_memory_size'] = len(self.embryo.memory)
        
        if self.is_learn():
            if self.learn_step % self.embryo.update_target_iter == 0:
                self.embryo.sync_update_weights()
            
            loss = self.embryo.update(self.learn_step)
            self.learn_step += 1
            self.alg_scalar_data['train_step_loss'] = loss

            return loss, self.learn_step
    
    def is_learn(self):
        return self.embryo.is_ready

    def policy(self, status, action_dim, explore_step=None):
        self.alg_scalar_data['train_epsilon_value'] = self.embryo.epsilon_scheduler.value
        if self.embryo.warmup_full_random and not self.is_learn() and explore_step is None:
            return function(np.random.randint, action_dim)()
        else:
            explore_step = self.learn_step if explore_step is None else explore_step
            return self.embryo.epsilon_scheduler.explore(explore_step, function(np.random.randint, action_dim), function(self.predict, status, action_dim))
    
    def save(self, save_path, use_full_path=False):
        save_path = str(save_path) if use_full_path else str(save_path / (self.__class__.__name__ + '.pt'))
        super().save(save_path, self.embryo.policy_model)