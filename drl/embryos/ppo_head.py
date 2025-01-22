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
        batch_state, batch_action, batch_reward, batch_next_state, batch_terminal, other_args = self.get_memory()
        batch_state = torch.tensor(batch_state, dtype=torch.float).squeeze()
        batch_action = torch.tensor(batch_action, dtype=torch.long).squeeze()
        batch_reward = torch.tensor(batch_reward, dtype=torch.float).squeeze()
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float).squeeze()
        batch_terminal = torch.tensor(batch_terminal, dtype=torch.bool).squeeze()
        batch_old_probs = torch.tensor(other_args[1], dtype=torch.float).squeeze()
        batch_old_value = torch.tensor(other_args[0], dtype=torch.float).squeeze()
        
        batch_adv = torch.zeros(self.batch_size, self.step_len, dtype=torch.float32)  
        batch_target_value = torch.zeros(self.batch_size, self.step_len, dtype=torch.float32)  
        for bt_id in range(self.batch_size):
            reward = batch_reward[bt_id].detach()
            done = batch_terminal[bt_id].detach()
            value = batch_old_value[bt_id].detach()

            _, next_v = self.policy_model.forward(batch_next_state[bt_id])
            value = torch.cat((value, next_v.detach()[-1]))   # value add last_value
            next_value = value[1:]
            value = value[:-1]
            discount = ~done * self.gamma
            delta_t = reward + discount * next_value - value
            adv = delta_t

            for j in range(len(adv) - 2, -1, -1):
                adv[j] += adv[j + 1] * discount[j] * self.gae_lambda

            target_value = adv + value

            batch_adv[bt_id] = adv
            batch_target_value[bt_id] = target_value
        
        # reshape
        batch_state = batch_state
        batch_action = batch_action
        batch_old_probs = batch_old_probs
        batch_old_value = batch_old_value
        clipped_len = self.step_len 

        if len(batch_state.shape) > 3:
            # atari
            batch_state = batch_state.reshape(self.batch_size * clipped_len, batch_state.shape[-3], batch_state.shape[-2], batch_state.shape[-1])
        else:
            batch_state = batch_state.reshape(self.batch_size * clipped_len, batch_state.shape[-1])
        batch_action = batch_action.reshape(self.batch_size * clipped_len)
        batch_old_probs = batch_old_probs.reshape(self.batch_size * clipped_len)
        batch_old_values = batch_old_value.reshape(self.batch_size * clipped_len)

        batch_adv = batch_adv.reshape(self.batch_size * clipped_len)
        batch_target_value = batch_target_value.reshape(self.batch_size * clipped_len)

        update_loss = 0
        mini_batch = 200
        nbatch = self.batch_size * clipped_len
        inds = np.arange(nbatch)
        np.random.shuffle(inds)
        for _ in range(self.update_step):
            for start in range(0, nbatch, mini_batch):
                end = start + mini_batch
                mbinds = inds[start:end]
                states = batch_state[mbinds]
                actions = batch_action[mbinds]
                old_probs = batch_old_probs[mbinds]
                advs = batch_adv[mbinds]
                target_values = batch_target_value[mbinds]
                old_values = batch_old_values[mbinds]

                action_new_probs, v = self.policy_model.forward(states)
                v = v.squeeze()
                action_dist = torch.distributions.Categorical(action_new_probs.squeeze())
                action_log_prob = action_dist.log_prob(actions)
                action_dist_entropy = action_dist.entropy()

                ratio = torch.exp(action_log_prob - old_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advs
                
                vf_losses1 = torch.pow(v - target_values, 2)
                val_pred_clipped = old_values + torch.clamp(v - old_values, min=-5, max=5)
                vf_losses2 = torch.pow(val_pred_clipped - target_values, 2)
                critic_loss = 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))

                loss = -torch.min(surr1, surr2) + critic_loss - action_dist_entropy * 0.01
                update_loss += loss.mean().detach()

                self.optimizer.zero_grad()
                loss.mean().backward(retain_graph=True)
                self.optimizer.step()

        self.policy_model.sync_weights_to(self.target_model)
        return update_loss / self.update_step

    def execute(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state).squeeze(), 0)
        action_prob, step_v_out = self.policy_model.forward(state)
        action_dist = torch.distributions.Categorical(action_prob.detach().squeeze())
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        return action, action_log_prob, step_v_out.detach()

    def save_memory(self, *args):
        assert self.memory is not None
        self.memory.push(*args)

    def get_memory(self):
        assert self.memory is not None
        return self.memory.convert(self.memory.pop(self.batch_size))  # used but dropped
        # return self.memory.convert(
        #     random.sample(list(self.memory), self.batch_size)
        # )  # used but saved

    def clear_memory(self):
        assert self.memory is not None
        self.memory.clear()
