# -*- coding: utf-8 -*-
from drl.utils.epsilon_schedulers.scheduler import EpsilonScheduler



__all__ = ['ConstantEpsilonScheduler']


class ConstantEpsilonScheduler(EpsilonScheduler):
    def __init__(self, epsilon_value):
        assert 0 <= epsilon_value <= 1
        self.epsilon_value = epsilon_value
    
    def step(self, x):
        pass
    
    def explore(self, x, random_action_func, greedy_action_func):
        action, _ = self.epsilon_greedy(self.epsilon_value, random_action_func, greedy_action_func)
        return action