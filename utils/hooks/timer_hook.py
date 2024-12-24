# -*- coding: utf-8 -*-
import time

from utils.hook import Hook, HOOKS
from utils.parallel.distributed import dispath_process_func


__all__ = ['TimerHook']


@HOOKS.register_module()
class TimerHook(Hook):
    
    @dispath_process_func()
    def before_run(self, enginer):
        self.t = time.time()

    @dispath_process_func()
    def after_train_episode(self, enginer):
        episode_time = time.time() - self.t
        time_loss_scalar = (0, 0) if hasattr(enginer, "episode_loss") else (episode_time, sum(enginer.episode_loss) / (len(enginer.episode_loss) + 1e-3))
        enginer.scalar_buffer.update({'time_step': (episode_time, enginer.episode_step),
                                     'time_reward': (episode_time, enginer.episode_reward),
                                     'time_loss': time_loss_scalar})

    @dispath_process_func()
    def after_run(self, enginer):
        self.t = time.time()