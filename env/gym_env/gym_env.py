# -*- coding: utf-8 -*-
import gym
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple

from env.base_env import BaseEnv
from env.builder import ENV

__all__ = ["GymEnv"]


@ENV.register_module()
class GymEnv(BaseEnv):
    def __init__(
        self,
        gym_name: str,
        *args,
        render: bool = False,
        wrappers: list = None,
        **kwargs,
    ):
        super(GymEnv, self).__init__()
        self.name = gym_name
        self.is_render = render
        self.wrappers = wrappers
        self.init()

    def init(self):
        try:
            if self.wrappers and isinstance(self.wrappers, list):
                from env.gym_env.wrapper import wrapper_table

                self.env = gym.make(self.name)
                for wrapper_kwargs in self.wrappers:
                    wrapper_type = wrapper_kwargs.pop("type")
                    self.env = wrapper_table[wrapper_type](self.env, **wrapper_kwargs)
            else:
                self.env = gym.make(self.name).unwrapped
        except:
            raise AttributeError(
                f"game '{self.name}' maybe not install, please check it."
            )

        if isinstance(self.env.action_space, Box):
            self._action_dim = self.env.action_space.shape
        elif isinstance(self.env.action_space, Discrete):
            self._action_dim = self.env.action_space.n

        if isinstance(self.env.observation_space, Box):
            self._status_dim = self.env.observation_space.shape
        elif isinstance(self.env.observation_space, Discrete):
            self._status_dim = self.env.observation_space.n
        else:
            raise AttributeError("gym state type need adaptation")

    def __repr__(self):
        return f"env name: {self.name}, state dim: {self._status_dim}, action dim: {self._action_dim}"

    def _get_all_gym_env(self):
        from gym import envs

        print(envs.registry.all())

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def status_dim(self):
        return self._status_dim

    def step(self, action):
        return self.env.step(action)

    def render(self):
        if self.is_render:
            self.env.render()

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed=seed)

    def atari_lives_end(self):
        return self.env.unwrapped.ale.lives() == 0
