# -*- coding: utf-8 -*-
import inspect
import torch
import numpy as np
from drl.utils.explores.vibrate import Vibrate



__all__ = ['OrnsteinUhlenbeckVibrateExplore']


class OrnsteinUhlenbeckVibrateExplore(Vibrate):

    def __init__(self, mean=0.0, theta=0.15, sigma=0.3):
        self.mean = mean
        self.theta = theta
        self.sigma = sigma
        self.ou_value = None

    def sample(self, a):
        a = a() if inspect.isfunction(a) else a

        # 确保 a 是 NumPy 数组或 PyTorch 张量
        if isinstance(a, np.ndarray):
            pass  # 如果已经是 NumPy 数组，无需处理
        elif isinstance(a, torch.Tensor):
            a = a.numpy()  # 将 PyTorch 张量转换为 NumPy 数组
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
        
        if self.ou_value is None:
            std = self.sigma / np.sqrt(2 * self.theta - self.theta ** 2)
            self.ou_value = np.random.normal(loc=self.mean, scale=std, size=a.shape).astype(np.float32)
        else:
            w = np.random.normal(loc=self.mean, scale=self.sigma, size=self.ou_value.shape)
            self.ou_value += self.theta * (self.mean - self.ou_value) + w
        
        noise = self.ou_value
        return a + noise
