# -*- coding: utf-8 -*-


__all__ = ['ModelBase']



class ModelBase(object):
    def __init__(self):
        ...

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_weights(self, *args, **kwargs):
        raise NotImplementedError

    def set_weights(self, weights, *args, **kwargs):
        raise NotImplementedError

    def sync_weights_to(self, target_model):
        """
        Copy paramter to target model
        """
        raise NotImplementedError
    
    def parameters(self):
        """
        Get the parameters of the model.
        """
        raise NotImplementedError

    def init_params(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)