# -*- coding: utf-8 -*-
import torch
import numpy as np
from drl.utils.explores.vibrate import Vibrate



__all__ = ['GaussianVibrateExplore']


class GaussianVibrateExplore(Vibrate):

    def __init__(self, mean=0.0, std=1e-2, low=None, high=None):
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high

    def sample(self, a, to_numpy=False):
        a = a() if callable(a) else a
        # a = torch.from_numpy(a) if isinstance(a, np.ndarray) else a
        
        # 确保 a 是 NumPy 数组或 PyTorch 张量
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)  # 将 NumPy 数组转换为 PyTorch 张量
        elif isinstance(a, torch.Tensor):
            pass  # 如果已经是 PyTorch 张量，无需处理
        else:
            raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
        
        noise = torch.normal(mean=self.mean, std=self.std, size=a.shape)
        if self.low is not None or self.high is not None:
            sample_data = torch.clamp(a + noise, self.low, self.high)
        else:
            sample_data = a + noise
        
        if to_numpy:
            return sample_data.numpy()
        return sample_data