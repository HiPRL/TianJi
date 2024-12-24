# -*- coding: utf-8 -*-
import torch.nn as nn
from drl.base.model import Model
from drl.builder import MODELS
from drl.models.utils import *

__all__ = ["PPOActorCritic"]


@MODELS.register_module()
class PPOActorCritic(Model):
    def __init__(
        self,
        actor_state_dim,
        actor_action_dim,
        critic_state_dim,
        mlp_dim=256,
        act: str = None,
        init_method: str = "kaiming_init",
        **kwargs
    ):
        super(PPOActorCritic, self).__init__(init_method=init_method, **kwargs)
        self.actor = Actor(
            actor_state_dim,
            actor_action_dim,
            mlp_dim=mlp_dim,
            act=act,
            init_method=init_method,
            **kwargs
        )
        self.critic = Critic(
            critic_state_dim,
            mlp_dim=mlp_dim,
            act=act,
            init_method=init_method,
            **kwargs
        )

    def init_params(self):
        self.actor.init_params()
        self.critic.init_params()

    def actor_forward(self, states):
        return self.actor(states)

    def critic_forward(self, states):
        return self.critic(states)


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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.act(self.fc1(state))
        x = self.act(self.fc2(x))
        model_out = self.softmax(self.out(x))
        return model_out


class Critic(Model):
    def __init__(
        self, state_dim, mlp_dim=256, act: str = None, init_method: str = None, **kwargs
    ):
        super(Critic, self).__init__(init_method=init_method, **kwargs)
        self.linear_c1 = nn.Linear(state_dim, mlp_dim)
        self.linear_c2 = nn.Linear(mlp_dim, mlp_dim)
        self.linear_out = nn.Linear(mlp_dim, 1)
        self.act = nn.ReLU() if act is None else activation_insidious_table[act.lower()]

    def forward(self, states):
        x = self.act(self.linear_c1(states))
        x = self.act(self.linear_c2(x))
        return self.linear_out(x)
