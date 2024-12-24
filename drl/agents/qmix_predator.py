# -*- coding: utf-8 -*-
import torch
import numpy as np
from utils.common import function

from drl.builder import AGENTS
from drl.base.agent import Agent



__all__ = ['PredatorQMIX']



@AGENTS.register_module()
class PredatorQMIX(Agent):
    def __init__(self, embryo):
        super(PredatorQMIX, self).__init__(embryo)
    
    def learn(self, *args, **kwargs):
        if args:
            self.embryo.save_memory(*args)
            self.alg_scalar_data['train_memory_size'] = len(self.embryo.memory)
        
        if self.is_learn():
            if self.learn_step % self.embryo.update_target_iter == 0:
                self.embryo.sync_update_weights()

            loss, _ = self.embryo.update()
            self.learn_step += 1
            self.alg_scalar_data['train_step_loss'] = loss

            return loss, self.learn_step
    
    def is_learn(self):
        return self.embryo.is_ready
    
    def predict(self, agents_obs, _):
        pred_Q = self.embryo.execute(np.array(agents_obs)).cpu().detach().squeeze()
        actions = pred_Q.max(dim=1)[1].cpu().numpy()
        return actions
    
    def policy(self, agents_obs, agents_action, explore_step=None):
        callback = lambda x : x.sample().long().cpu().detach().numpy()
        random_action_func = function(torch.distributions.Categorical, torch.tensor(agents_action, dtype=torch.float32), callback=callback)
        self.alg_scalar_data['train_epsilon_value'] = self.embryo.epsilon_scheduler.value
        if self.embryo.warmup_full_random and not self.is_learn() and explore_step is None:
            return random_action_func()
        else:
            explore_step = self.learn_step if explore_step is None else explore_step
            return self.embryo.epsilon_scheduler.explore(explore_step, random_action_func, function(self.predict, agents_obs, None))

    def update_model_params(self, weights):
        """
        need to update
        """
        self.embryo.set_weights(weights)
    
    def take_model_params(self):
        """
        need to update
        """
        return self.embryo.get_weights()

    def save(self, save_path, use_full_path=False):
        save_path = str(save_path) if use_full_path else str(save_path / (self.__class__.__name__ + '.pt'))
        model_dict = {'model':self.embryo.policy_model, 'mixer_model':self.embryo.policy_mixer_model}
        super().save(save_path, model_dict)
    
    def resume(self, save_path):
        model_dict = torch.load(save_path)
        self.embryo.policy_model.load_state_dict(model_dict['model'].state_dict())
        self.embryo.policy_mixer_model.load_state_dict(model_dict['mixer_model'].state_dict())