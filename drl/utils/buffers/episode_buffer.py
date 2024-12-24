# -*- coding: utf-8 -*-
import numpy as np
from drl.utils.buffers import BUFFER, Experience, ReplayMemory



__all__ = ['EpisodeBuffer']



@BUFFER.register_module()
class EpisodeBuffer(ReplayMemory):
    def __init__(self, max_size, episode_limit=None):
        self.episode_len = episode_limit
        self.push_flag = False
        self.cur_episode = 0
        self.cur_episode_index = 0
        super(EpisodeBuffer, self).__init__(max_size)
    
    def push(self, *args, force=False):
        if force:
            # EpisodeBuffer force push must be a complete episode data
            self.push_flag = True
            super().push(*args, force=force)
        else:
            data = Experience(*args)
            assert (data.index >= self.cur_episode), f"episode index must be more than cur episode index {self.cur_episode}, but got {data.index}"
            if self.cur_episode_index == 0:
                self._init_episode(data)
            
            if data.index == self.cur_episode:
                if self.episode_len:
                    self.episode_state[self.cur_episode_index] = data.state
                    self.episode_action[self.cur_episode_index] = data.action
                    self.episode_reward[self.cur_episode_index] = data.reward
                    self.episode_next_state[self.cur_episode_index] = data.next_state
                    self.episode_terminal[self.cur_episode_index] = data.terminal
                    for i, item in enumerate(data.other_args):
                        self.episode_other_args[i][self.cur_episode_index] = item
                else:
                    self.episode_state.append(data.state)
                    self.episode_action.append(data.action)
                    self.episode_reward.append(data.reward)
                    self.episode_next_state.append(data.next_state)
                    self.episode_terminal.append(data.terminal)
                    for i, item in enumerate(data.other_args):
                        self.episode_other_args[i].append(item)
                self.cur_episode_index += 1
                self.push_flag = False
            else:
                if self.episode_len is None:
                    self.episode_state = np.array(np.vstack(self.episode_state))
                    self.episode_action = np.array(np.vstack(self.episode_action))
                    self.episode_reward = np.array(np.vstack(self.episode_reward))
                    self.episode_next_state = np.array(np.vstack(self.episode_next_state))
                    self.episode_terminal = np.array(np.vstack(self.episode_terminal))
                    self.episode_other_args = [np.array(np.vstack(item)) for item in self.episode_other_args]
                self.pool.append(Experience(self.cur_episode, self.episode_state, self.episode_action, self.episode_reward, self.episode_next_state, self.episode_terminal, self.episode_other_args))
                self.next_idx = (self.next_idx + 1) % self.max_size
                self.cur_episode = data.index
                self.cur_episode_index = 0
                self.push_flag = True

    @property
    def push_status(self):
        return self.push_flag

    def _init_episode(self, data):
        if self.episode_len:
            self.episode_state = np.zeros((self.episode_len,) + np.array(data.state).squeeze().shape)
            self.episode_action = np.zeros((self.episode_len,) + np.array(data.action).squeeze().shape)
            self.episode_reward = np.zeros((self.episode_len,) + np.array(data.reward).squeeze().shape)
            self.episode_next_state = np.zeros((self.episode_len,) + np.array(data.next_state).squeeze().shape)
            self.episode_terminal = np.ones((self.episode_len,) + np.array(data.terminal).squeeze().shape)
            self.episode_other_args = [np.zeros((self.episode_len,) + np.array(item).squeeze().shape) for item in data.other_args]
        else:
            self.episode_state = []
            self.episode_action = []
            self.episode_reward = []
            self.episode_next_state = []
            self.episode_terminal = []
            self.episode_other_args = [[] for _ in data.other_args]

    def sample_batch(self, batch_size):
        data = Experience(*zip(*self._sample_batch(batch_size)))
        if self.episode_len:
            batch_state = np.array(data.state)
            batch_action = np.array(data.action)
            batch_reward = np.array(data.reward)
            batch_next_state = np.array(data.next_state)
            batch_terminal = np.array(data.terminal)
            args_dict = {}
            for data in data.other_args:
                for index, item in enumerate(data if isinstance(data, list) else [data]):
                    args_dict.setdefault(index, []).append(item)
            batch_other_args = tuple(map(lambda item: np.asarray(item), args_dict.values()))
        else:
            batch_state = data.state
            batch_action = data.action
            batch_reward = data.reward
            batch_next_state = data.next_state
            batch_terminal = data.terminal
            args_dict = {}
            for data in data.other_args:
                for index, item in enumerate(data if isinstance(data, list) else [data]):
                    args_dict.setdefault(index, []).append(item)
            batch_other_args = tuple(args_dict.values())
        return batch_state, batch_action, batch_reward, batch_next_state, batch_terminal, batch_other_args