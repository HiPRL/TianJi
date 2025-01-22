# -*- coding: utf-8 -*-
import numpy as np
from drl.utils.buffers import BUFFER, Experience, ReplayMemory



__all__ = ['MultiStepBuffer']



@BUFFER.register_module()
class MultiStepBuffer(ReplayMemory):
    def __init__(self, max_size, step_limit):
        self.step_len = step_limit
        self.push_flag = False
        self.cur_step = 0
        self.cur_step_index = 0
        super(MultiStepBuffer, self).__init__(max_size)
    
    def push(self, *args, force=False):
        if force:
            # MultiStepBuffer force push must be a complete step data for distribute training.
            super().push(*args, force=force)
        else:
            data = Experience(*args)
            if self.cur_step_index % self.step_len == 0:
                self.push_flag = False
                self._init_step(data)
            
            if self.cur_step < self.step_len:
                self.step_state[self.cur_step] = data.state.squeeze()
                self.step_action[self.cur_step] = data.action
                self.step_reward[self.cur_step] = data.reward
                self.step_next_state[self.cur_step] = data.next_state.squeeze()
                self.step_terminal[self.cur_step] = data.terminal
                # self.step_value[self.cur_step] = data.value
                for i, item in enumerate(data.other_args):
                    self.step_other_args[i][self.cur_step] = item
                self.cur_step += 1
                self.cur_step_index += 1
        
            if self.cur_step_index % self.step_len == 0:
                self.cur_step = 0
                self.push_flag = True
            
            if self.push_flag:
                self.pool.append(Experience(self.cur_step_index, self.step_state, self.step_action, self.step_reward, self.step_next_state, self.step_terminal, self.step_other_args))

    def _init_step(self, data):
        self.step_state = np.zeros((self.step_len,) + np.array(data.state).squeeze().shape, dtype=np.float32)
        self.step_action = np.zeros((self.step_len,) + np.array(data.action).squeeze().shape, dtype=np.float32)
        self.step_reward = np.zeros((self.step_len,) + np.array(data.reward).squeeze().shape, dtype=np.float32)
        self.step_next_state = np.zeros((self.step_len,) + np.array(data.next_state).squeeze().shape, dtype=np.float32)
        self.step_terminal = np.zeros((self.step_len,) + np.array(data.terminal).squeeze().shape, dtype=np.float32)
        # self.step_value = np.zeros((self.step_len,) + np.array(data.value).squeeze().shape, dtype=np.float32)
        self.step_other_args = [np.zeros((self.step_len,) + np.array(item).squeeze().shape, dtype=np.float32) for item in data.other_args]

    def sample_batch(self, batch_size):
        data = Experience(*zip(*self._sample_batch(batch_size)))
        batch_state = np.array(data.state)
        batch_action = np.array(data.action)
        batch_reward = np.array(data.reward)
        batch_next_state = np.array(data.next_state)
        batch_terminal = np.array(data.terminal)
        # batch_value = np.array(data.value)
        args_dict = {}
        for data in data.other_args:
            for index, item in enumerate(data if isinstance(data, list) else [data]):
                args_dict.setdefault(index, []).append(item)
        batch_other_args = tuple(map(lambda item: np.asarray(item), args_dict.values()))
        return batch_state, batch_action, batch_reward, batch_next_state, batch_terminal, batch_other_args

    def convert(self, data):
        batch_size, data = len(data), Experience(*zip(*data))
        batch_state = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='float32'), data.state))).reshape((batch_size, ) + data.state[0].shape).squeeze()
        batch_action = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='float32'), data.action)))
        batch_reward = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='float32'), data.reward)))
        batch_next_state = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='float32'), data.next_state))).reshape((batch_size, ) + data.next_state[0].shape).squeeze()
        batch_terminal = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='bool'), data.terminal)))
        # batch_value = np.vstack(tuple(map(lambda item: np.asarray(item, dtype='float32'), data.value)))
        args_dict = {}
        for data in data.other_args:
            for index, item in enumerate(data if isinstance(data, list) else [data]):
                args_dict.setdefault(index, []).append(item)
        batch_other_args = tuple(map(lambda item: np.asarray(item), args_dict.values()))
        return batch_state, batch_action, batch_reward, batch_next_state, batch_terminal, batch_other_args