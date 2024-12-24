# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict


__all__ = ['ScalarBuffer']

class ScalarBuffer:
    def __init__(self):
        self.buffer = OrderedDict()
    
    def update(self, scalar, index=None):
        if scalar:
            if index:
                assert isinstance(index, (int, float))
                for key, value in scalar.items():
                    self.buffer.setdefault(key, []).append((index, value))
            else:
                for key, value in scalar.items():
                    self.buffer.setdefault(key, []).append(value)

    def data(self):
        for name, val in self.buffer.items():
            for item in val:
                yield name, np.array(item[1]), item[0]

    def clear(self):
        self.buffer.clear()