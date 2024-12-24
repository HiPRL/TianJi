# -*- coding: utf-8 -*-
import os.path as osp
from torch.utils.tensorboard import SummaryWriter

from utils.hook import Hook, HOOKS
from utils.parallel.distributed import dispath_process_func


__all__ = ['TensorboardHook']


@HOOKS.register_module()
class TensorboardHook(Hook):
    
    @dispath_process_func()
    def before_run(self, enginer):
        print("Start with 'tensorboard --logdir [path]', view at http://localhost:6006/.", flush=True)
        self.scalar_writer = SummaryWriter(osp.join(enginer.save_dir, 'scalar'))

    @dispath_process_func()
    def after_train_episode(self, enginer):
        for data in enginer.scalar_buffer.data():
            self.scalar_writer.add_scalar(*data)
        enginer.scalar_buffer.clear()

    @dispath_process_func()
    def after_run(self, enginer):
        self.scalar_writer.flush()
        self.scalar_writer.close()