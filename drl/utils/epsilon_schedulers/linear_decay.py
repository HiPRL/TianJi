# -*- coding: utf-8 -*-
from drl.utils.epsilon_schedulers.scheduler import EpsilonScheduler



__all__ = ['LinearDecayEpsilonScheduler']


class LinearDecayEpsilonScheduler(EpsilonScheduler):
    def __init__(self, start_epsilon, end_epsilon, max_step, min_step=1):
        assert 0 <= start_epsilon <= 1
        assert 0 <= end_epsilon <= 1
        assert 0 <= min_step <= max_step
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.max_step = max_step
        self.min_step = min_step
        self.epsilon_value = 0
    
    def step(self, x):
        if x < self.min_step:
            return self.start_epsilon
        if x > self.max_step:
            return self.end_epsilon
        return self.start_epsilon + (self.end_epsilon - self.start_epsilon) * ((x - self.min_step) / (self.max_step - self.min_step))
    
    def explore(self, x, random_action_func, greedy_action_func):
        self.epsilon_value = self.step(x)
        action, _ = self.epsilon_greedy(self.epsilon_value, random_action_func, greedy_action_func)
        return action
    
    @property
    def value(self):
        return self.epsilon_value