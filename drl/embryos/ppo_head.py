# -*- coding: utf-8 -*-
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from drl.base.embryo import Embryo
from drl.builder import EMBRYOS
from drl.utils.buffers import MultiStepBuffer

__all__ = ["PPOHead"]


@EMBRYOS.register_module()
class PPOHead(Embryo):
    def __init__(self, model, hyp):
        super(PPOHead, self).__init__(model)
        self.policy_model = model
        self.target_model = copy.deepcopy(model)
        self.lr = hyp.lr
        self.clip_param = hyp.clip_param
        self.gamma = hyp.gamma
        self.gae_lambda = hyp.gae_lambda
        self.update_step = hyp.update_step
        self.batch_size = hyp.batch_size
        self.step_len = hyp.step_len
        self.buffer_size = hyp.buffer_size
        self.warmup_size = hyp.get("warmup_size", None)
        self.warmup_full_random = hyp.get("warmup_full_random", None)

        self.memory = MultiStepBuffer(self.buffer_size, hyp.step_len)
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.policy_model.parameters(), lr=self.lr, eps=1e-5
        )

    def update(self, *args, **kwargs):
        (
            batch_state,
            batch_action,
            batch_reward,
            batch_next_state,
            batch_terminal,
            probs,
        ) = self.get_memory()
        batch_state = torch.tensor(batch_state, dtype=torch.float).to(self.sync_device)
        batch_action = torch.tensor(batch_action, dtype=torch.long).to(self.sync_device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float).to(self.sync_device)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float).to(
            self.sync_device
        )
        batch_terminal = torch.tensor(batch_terminal, dtype=torch.bool).to(self.sync_device)
        batch_old_probs = torch.tensor(probs[0], dtype=torch.float).to(self.sync_device)

        batch_adg_reward = []
        for i, rewards in enumerate(batch_reward):
            adg_rewards = []
            discounted_reward = 0
            terminals = batch_terminal[i]
            for rw, is_terminal in zip(reversed(rewards), reversed(terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = rw + (self.gamma * discounted_reward)
                adg_rewards.insert(0, discounted_reward)
            batch_adg_reward.append(adg_rewards)

        nor_rewards = torch.tensor(batch_adg_reward, dtype=torch.float32).to(
            self.sync_device
        )
        nor_rewards = (nor_rewards - nor_rewards.mean()) / (nor_rewards.std() + 1e-7)
        old_state_values = (
            self.target_model.critic_forward(batch_state).squeeze().detach()
        )
        advantages = nor_rewards.detach() - old_state_values.detach()

        # reshape
        advantages = advantages.reshape(self.batch_size * self.step_len)
        batch_old_probs = batch_old_probs.reshape(self.batch_size * self.step_len)
        batch_state = batch_state.reshape(
            self.batch_size * self.step_len, batch_state.shape[-1]
        )
        batch_action = batch_action.reshape(self.batch_size * self.step_len)
        nor_rewards = nor_rewards.reshape(self.batch_size * self.step_len)

        update_loss = 0
        for _ in range(self.update_step):
            action_new_probs = self.policy_model.actor_forward(batch_state)
            action_dist = torch.distributions.Categorical(action_new_probs.squeeze())
            action_log_prob = action_dist.log_prob(batch_action)
            action_dist_entropy = action_dist.entropy()
            v = self.policy_model.critic_forward(batch_state)

            ratio = torch.exp(action_log_prob - batch_old_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                * advantages
            )
            loss = (
                -torch.min(surr1, surr2)
                + self.mse_loss(v.squeeze(), nor_rewards) * 0.5
                - action_dist_entropy * 0.01
            )
            update_loss += loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_model.sync_weights_to(self.target_model)
        return update_loss / self.update_step

    def execute(self, state, use_target_model=False):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.sync_device)
        action_prob = (
            self.target_model.actor_forward(state)
            if use_target_model
            else self.policy_model.actor_forward(state)
        )
        action_dist = torch.distributions.Categorical(action_prob.squeeze())
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action).sum(-1, keepdim=True)
        return action, action_log_prob

    def save_memory(self, *args):
        assert self.memory is not None
        self.memory.push(*args)

    def get_memory(self):
        assert self.memory is not None
        # return self.memory.convert(self.memory.pop(self.batch_size))  # used but dropped
        return self.memory.convert(
            random.sample(list(self.memory), self.batch_size)
        )  # used but saved

    def clear_memory(self):
        assert self.memory is not None
        self.memory.clear()
