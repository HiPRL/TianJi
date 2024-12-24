# -*- coding: utf-8 -*-


__all__ = ['AgentBase']



class AgentBase(object):
    def __init__(self):
        ...

    def learn(self, *args, **kwargs):
        raise NotImplementedError()
    
    def predict(self, *args, **kwargs):
        raise NotImplementedError()
    
    def policy(self, *args, **kwargs):
        raise NotImplementedError()