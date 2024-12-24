# -*- coding: utf-8 -*-
from utils import Registry



__all__ = ['ENV', 'build_env']



ENV = Registry('environment')

def build_env(cfg):
    return ENV.build(cfg)