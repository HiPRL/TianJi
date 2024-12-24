# -*- coding: utf-8 -*-
import os

os.environ["SC2PATH"] = "/home/pdl_lpl/desktop/mnt/pkgs/StarCraftII"

from collections import OrderedDict

import numpy as np
from env.base_env import BaseEnv
from env.builder import ENV
from smac.env import StarCraft2Env as sc2
from smac.env.starcraft2.maps.smac_maps import map_param_registry

__all__ = ["StarCraft2Env"]


class OneHotTransform(object):
    def __init__(self, data_dim):
        self.data_dim = data_dim

    def __call__(self, agent_id):
        assert agent_id < self.data_dim
        one_hot_array = np.zeros(self.data_dim, dtype="float32")
        one_hot_array[agent_id] = 1.0
        return one_hot_array


@ENV.register_module()
class StarCraft2Env(BaseEnv):
    """
    Secondary packaging The StarCraft II environment for decentralised multi-agent micromanagement scenarios.
    """

    def __init__(self, map_name, *args, render=False, **kwargs):
        super(StarCraft2Env, self).__init__()
        self.map_name = map_name
        self.is_render = render
        self.init(*args, **kwargs)

    def init(self, *args, **kwargs):
        try:
            if self.map_name in map_param_registry.keys():
                self.env = sc2(self.map_name, *args, **kwargs)
                self.agent_id_onehot_func = OneHotTransform(self.env.n_agents)
                self.action_onehot_func = OneHotTransform(self.env.get_total_actions())
                agents_onehot = []
                for agent_id in range(self.env.n_agents):
                    one_hot = self.agent_id_onehot_func(agent_id)
                    agents_onehot.append(one_hot)
                self.agents_onehot = np.array(agents_onehot)
            else:
                raise IOError(
                    f'map name not registry in StarCraft2Env.\none option is to choose from the current registration map_name{tuple(map_param_registry.keys())}.\nthe other is to register a new map by yourself use "registry_map" function and make a SC2Map, for more information, see "https://github.com/oxwhirl/smac#smac-maps"'
                )
        except Exception as e:
            print(
                "Make StarCraft2Env environment warning, please debug if necessary for details."
            )

    @property
    def agent_num(self):
        return self.env.n_agents

    @property
    def enemy_num(self):
        return self.env.n_enemies

    @property
    def status_dim(self):
        return self.env.get_state_size()

    @property
    def observations_dim(self):
        return self.env.get_obs_size()

    @property
    def action_dim(self):
        return self.env.get_total_actions()

    @property
    def episode_limit(self):
        return self.env.episode_limit

    @property
    def win_status(self):
        return self.env.win_counted

    def step(self, actions):
        reward, terminated, info = self.env.step(actions)
        next_state = np.array(self.env.get_state())
        obs = np.array(self.env.get_obs())
        actions_one_hot = []
        for action in actions:
            one_hot = self.action_onehot_func(action)
            actions_one_hot.append(one_hot)
        actions_one_hot = np.array(actions_one_hot)
        self.obs = np.concatenate([obs, actions_one_hot, self.agents_onehot], axis=-1)
        return next_state, reward, terminated, info

    def local_observations(self):
        return self.obs

    def agents_avail_actions(self):
        available_actions = []
        for agent_id in range(self.agent_num):
            available_actions.append(self.env.get_avail_agent_actions(agent_id))
        return np.array(available_actions)

    def render(self):
        if self.is_render:
            self.env.render()

    def reset(self):
        observations, state = self.env.reset()
        last_actions_one_hot = np.zeros(
            (self.agent_num, self.action_dim), dtype="float32"
        )
        self.obs = np.concatenate(
            [observations, last_actions_one_hot, self.agents_onehot], axis=-1
        )
        return state

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env._seed = seed


def registry_map(name, data):
    assert isinstance(name, str), f"name type must be str, but got {type(name)}"
    assert name not in map_param_registry.keys(), "Duplicate name, please change name."
    smac_map = OrderedDict()
    smac_map["n_agents"] = data.n_agents
    smac_map["n_enemies"] = data.n_enemies
    smac_map["limit"] = data.limit
    smac_map["a_race"] = data.a_race
    smac_map["b_race"] = data.b_race
    smac_map["unit_type_bits"] = data.unit_type_bits
    smac_map["map_type"] = data.map_type
    map_param_registry[name] = smac_map
