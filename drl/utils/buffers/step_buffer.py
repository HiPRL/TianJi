# -*- coding: utf-8 -*-
import numpy as np
from drl.utils.buffers import BUFFER, Experience, ReplayMemory



__all__ = ['StepBuffer']



@BUFFER.register_module()
class StepBuffer(ReplayMemory):
    def __init__(self, max_size):
        super(StepBuffer, self).__init__(max_size)
    
    def sample_batch(self, batch_size):
        data = Experience(*zip(*self._sample_batch(batch_size)))
        batch_state = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='float64'), data.state))).reshape((batch_size, ) + data.state[0].shape).squeeze()
        batch_action = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='float64'), data.action)))
        batch_reward = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='float64'), data.reward)))
        batch_next_state = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='float64'), data.next_state))).reshape((batch_size, ) + data.next_state[0].shape).squeeze()
        batch_terminal = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='bool'), data.terminal)))
        args_dict = {}
        for data in data.other_args:
            for index, item in enumerate(data if isinstance(data, list) else [data]):
                args_dict.setdefault(index, []).append(item)
        batch_other_args = tuple(map(lambda item: np.asarray(item), args_dict.values()))
        return batch_state, batch_action, batch_reward, batch_next_state, batch_terminal, batch_other_args

    def convert(self, data):
        batch_size, data = len(data), Experience(*zip(*data))
        batch_state = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='float64'), data.state))).reshape((batch_size, ) + data.state[0].shape).squeeze()
        batch_action = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='float64'), data.action)))
        batch_reward = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='float64'), data.reward)))
        batch_next_state = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='float64'), data.next_state))).reshape((batch_size, ) + data.next_state[0].shape).squeeze()
        batch_terminal = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='bool'), data.terminal)))
        args_dict = {}
        for data in data.other_args:
            for index, item in enumerate(data if isinstance(data, list) else [data]):
                args_dict.setdefault(index, []).append(item)
        batch_other_args = tuple(map(lambda item: np.asarray(item), args_dict.values()))
        return batch_state, batch_action, batch_reward, batch_next_state, batch_terminal, batch_other_args