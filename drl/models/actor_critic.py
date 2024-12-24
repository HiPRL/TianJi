# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from drl.base.model import Model
from drl.builder import MODELS
from drl.models.utils import *

__all__ = ["ActorCritic"]


@MODELS.register_module()
class ActorCritic(Model):
    def __init__(
        self,
        actor_state_dim,
        actor_action_dim,
        critic_dim,
        mlp_dim=256,
        act: str = None,
        init_method: str = "kaiming_init",
        **kwargs
    ):
        super(ActorCritic, self).__init__(init_method=init_method, **kwargs)
        self.actor = Actor(
            actor_state_dim,
            actor_action_dim,
            mlp_dim=mlp_dim,
            act=act,
            init_method=init_method,
            **kwargs
        )
        self.critic = Critic(
            critic_dim, mlp_dim=mlp_dim, act=act, init_method=init_method, **kwargs
        )

    def init_params(self):
        self.actor.init_params()
        self.critic.init_params()

    def actor_forward(self, states, model_original_out=False):
        return self.actor(states, model_original_out)

    def critic_forward(self, states, actions):
        return self.critic(states, actions)


class Actor(Model):
    def __init__(
        self,
        state_dim,
        action_dim,
        mlp_dim=256,
        act: str = None,
        init_method: str = None,
        is_continuous_action: bool = False,
        **kwargs
    ):
        super(Actor, self).__init__(init_method=init_method, **kwargs)
        self.fc1 = nn.Linear(state_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.out = nn.Linear(mlp_dim, action_dim)
        self.act = nn.ReLU() if act is None else activation_insidious_table[act.lower()]
        self.out_act = nn.Tanh() if is_continuous_action else None

    def forward(self, state, model_original_out=False):
        x = self.act(self.fc1(state))
        x = self.act(self.fc2(x))
        model_out = self.out(x)

        # TODO 产生一个扰动的状态-动作分布, 可优化？扰动分布操作放置算法更新部分？
        policy = nn.functional.softmax(
            model_out - torch.log(-torch.log(torch.rand_like(model_out))), dim=-1
        )
        if model_original_out:
            return model_out, policy
        return policy


class Critic(Model):
    def __init__(
        self,
        critic_dim,
        mlp_dim=256,
        act: str = None,
        init_method: str = None,
        **kwargs
    ):
        super(Critic, self).__init__(init_method=init_method, **kwargs)
        self.linear_c1 = nn.Linear(critic_dim, mlp_dim)
        self.linear_c2 = nn.Linear(mlp_dim, mlp_dim)
        self.linear_out = nn.Linear(mlp_dim, 1)
        self.act = nn.ReLU() if act is None else activation_insidious_table[act.lower()]

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        x = self.act(self.linear_c1(x))
        x = self.act(self.linear_c2(x))
        return self.linear_out(x)
