# -*- coding: utf-8 -*-
from utils.hook import Hook, HOOKS


__all__ = ['DQNCartPoleV1Hook']


@HOOKS.register_module()
class DQNCartPoleV1Hook(Hook):
    def __init__(self):
        pass
    
    def before_train_step(self, enginer):
        return {"policy_factor": [enginer.status, enginer.env.action_dim], "learn_factor": [None]}

    def before_train_learn(self, enginer):
        if enginer.env.name == "CartPole-v1":
            try:
                # Convert discrete rewards to continuous rewards for CartPole-v1 environment.
                x, x_dot, theta, theta_dot = enginer.status
                reward1 = (enginer.env.env.x_threshold - abs(x)) / enginer.env.env.x_threshold - 0.8
                reward2 = (enginer.env.env.theta_threshold_radians - abs(theta)) / enginer.env.env.theta_threshold_radians - 0.5
                enginer.reward = reward1 + reward2
            except:
                pass
    
    def before_val_step(self, enginer):
        return {"policy_factor": [enginer.eval_status, enginer.eval_env.action_dim], "learn_factor": [None]}