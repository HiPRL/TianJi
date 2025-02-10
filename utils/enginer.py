# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import deepcopy

from utils.hook import HOOKS, Hook, build_from_cfg
from utils.priority import get_priority
from utils.scalar_buffer import ScalarBuffer

__all__ = ["Enginer"]


class Enginer(metaclass=ABCMeta):
    def __init__(self, agent, env, cfg):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.agent_num = 1 if not hasattr(cfg, "agent_num") else cfg.agent_num
        self.id = id(self)
        self.reward = None
        self.status = None
        self.eval_status = None
        self.eval_times = (
            1 if not hasattr(cfg.exp, "eval_times") else cfg.exp.eval_times
        )

        self.episode_step = 0
        self.episode_loss = None
        self.episode_reward = 0
        self.episode_test_step = 0
        self.episode_test_reward = 0

        self._hooks = []
        self._episode = 0
        self._train_step = 0

        # call_hook return
        self.__hook_rets = OrderedDict()

        self.save_dir = None if not hasattr(cfg, "save_dir") else cfg.save_dir
        self.scalar_buffer = ScalarBuffer()

    # Controlled by exp episodes(train episode num) and train_steps(step num)
    def train_flag(self):
        return (
            hasattr(self.cfg.exp, "episodes") and self._episode < self.cfg.exp.episodes
        ) or (
            hasattr(self.cfg.exp, "train_steps")
            and self._train_step < self.cfg.exp.train_steps
        )

    # Control a episode status
    def episode_flag(self, episode_step):
        return (
            hasattr(self.cfg.exp, "max_step") and episode_step > self.cfg.exp.max_step
        )

    # Control test condition
    def test_flag(self, steps=None):
        if not hasattr(self.cfg.exp, "eval_step"):
            return False

        if steps:
            return steps % self.cfg.exp.eval_step == 0 and not self.cfg.only_training
        else:
            return (
                self._episode % self.cfg.exp.eval_step == 0
                and not self.cfg.only_training
            )

    # Control model save condition
    def save_flag(self, steps=None):
        if not hasattr(self.cfg.exp, "save_freq"):
            return False

        if steps:
            return steps % self.cfg.exp.save_freq == 0 and self.cfg.is_save_model
        else:
            return (
                self._episode % self.cfg.exp.save_freq == 0 and self.cfg.is_save_model
            )

    def check_eval_env(self):
        if hasattr(self, "eval_env"):
            self.eval_env.close()
            del self.eval_env

        try:
            from env.builder import build_env

            self.eval_env = build_env(self.cfg.environment)
        except:
            self.eval_env = deepcopy(self.env)

    @abstractmethod
    def train_episode(self):
        pass

    @abstractmethod
    def test_episode(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def clone(self):
        try:
            obj = deepcopy(self)
        except:
            _cfg = deepcopy(self.cfg)

            try:
                from env.builder import build_env

                _env = build_env(_cfg.environment)
            except:
                _env = deepcopy(self.env)

            try:
                _agent = deepcopy(self.agent)
            except:
                from drl.builder import build_agent

                _agent = build_agent(_cfg.agent)

            obj = type(self)(_agent, _env, _cfg)
            obj.save_dir = self.save_dir
            obj.register_hook_from_cfg()

        if obj is None:
            raise IOError(f"The target {type(self)} does not support deep copy")
        return obj

    def register_hook(self, hook, priority="NORMAL"):
        assert isinstance(hook, Hook)
        if hasattr(hook, "priority"):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def register_hook_from_cfg(self):
        hook_cfg = [] if not hasattr(self.cfg, "hooks") else self.cfg.hooks.copy()
        hook_cfgs = hook_cfg if isinstance(hook_cfg, list) else [hook_cfg]
        if len(hook_cfgs) > 0:
            for item in hook_cfgs:
                priority = item.pop("priority", "NORMAL")
                hook = build_from_cfg(item, HOOKS)
                self.register_hook(hook, priority=priority)
        else:
            from drl.hooks import DefaultHook

            self.register_hook(DefaultHook())

    def call_hook(self, hook_name):
        self.__hook_rets.clear()
        for hook in self._hooks:
            ret = getattr(hook, hook_name)(self)
            if ret is not None:
                self.__hook_rets[hook.__class__.__name__] = ret
        if len(self.__hook_rets):
            return self.__hook_rets

    def reduce_factor(self, factor):
        if factor is None or not isinstance(factor, dict):
            return
        if len(factor) == 0:
            return
        for hook_ret in factor.values():
            if "policy_factor" in hook_ret.keys() and "learn_factor" in hook_ret.keys():
                return hook_ret["policy_factor"], hook_ret["learn_factor"]
