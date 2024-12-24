# -*- coding: utf-8 -*-
from utils import Registry


__all__ = ['Hook', 'HOOKS', 'build_from_cfg']



HOOKS = Registry('hook')
build_from_cfg = Registry.default_build


class Hook:

    def before_run(self, enginer):
        pass

    def after_run(self, enginer):
        pass

    def before_episode(self, enginer):
        pass

    def after_episode(self, enginer):
        pass

    def before_step(self, enginer):
        pass

    def after_step(self, enginer):
        pass

    def before_learn(self, enginer):
        pass

    def after_learn(self, enginer):
        pass

    def before_train_episode(self, enginer):
        self.before_episode(enginer)

    def before_val_episode(self, enginer):
        self.before_episode(enginer)

    def after_train_episode(self, enginer):
        self.after_episode(enginer)

    def after_val_episode(self, enginer):
        self.after_episode(enginer)

    def before_train_step(self, enginer):
        self.before_step(enginer)

    def before_val_step(self, enginer):
        self.before_step(enginer)

    def after_train_step(self, enginer):
        self.after_step(enginer)

    def after_val_step(self, enginer):
        self.after_step(enginer)

    def before_train_learn(self, enginer):
        self.before_learn(enginer)

    def after_train_learn(self, enginer):
        self.after_learn(enginer)