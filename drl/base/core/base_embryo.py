# -*- coding: utf-8 -*-


__all__ = ['EmbryoBase']



class EmbryoBase(object):
    def __init__(self):
        ...
    
    def update(self, *args, **kwargs):
        raise NotImplementedError()
    
    def execute(self, *args, **kwargs):
        raise NotImplementedError()