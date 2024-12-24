# -*- coding: utf-8 -*-
from abc import ABCMeta
from abc import abstractmethod



__all__ = ['BaseEnv']



class BaseEnv(object, metaclass=ABCMeta):
    @abstractmethod
    def init(self, *args, **kwargs):
        ...
    
    @abstractmethod
    def step(self, action):
        raise NotImplementedError()
    
    @abstractmethod
    def render(self):
        raise NotImplementedError()
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        raise NotImplementedError()
    
    def seed(self, seed):
        raise NotImplementedError()