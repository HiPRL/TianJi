# -*- coding: utf-8 -*-
from utils import Registry



__all__ = ['BUFFER', 'build_buffer']



BUFFER = Registry('buffer')

def build_buffer(cfg):
    return BUFFER.build(cfg)