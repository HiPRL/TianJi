# -*- coding: utf-8 -*-
import numpy as np




class EpsilonScheduler(object):

    def step(self, x):
        raise NotImplementedError()
    
    def explore(self, x, random_action_func, greedy_action_func):
        raise NotImplementedError()
    
    def epsilon_greedy(self, explore_value, random_action_func, greedy_action_func):
        if np.random.rand() < explore_value:
            return random_action_func(), False
        else:
            return greedy_action_func(), True