# -*- coding: utf-8 -*-
import copy

import torch
import torch.nn as nn
from drl.base.embryo import Embryo
from drl.builder import EMBRYOS
from drl.utils.epsilon_schedulers import LinearDecayEpsilonScheduler

__all__ = ["PredatorMixerHead"]


@EMBRYOS.register_module()
class PredatorMixerHead(Embryo):
    def __init__(self, model, mixer_model, hyp):
        super(PredatorMixerHead, self).__init__(model)
        self.policy_model = model
        self.target_model = copy.deepcopy(model)

        self.policy_mixer_model = mixer_model
        self.target_mixer_model = copy.deepcopy(mixer_model)

        self.lr = hyp.LR
        self.gamma = hyp.GAMMA
        self.update_target_iter = hyp.TARGET_REPLACE_ITER
        self.batch_size = hyp.batch_size
        self.buffer_size = hyp.buffer_size
        self.warmup_size = hyp.get("warmup_size", None)
        self.warmup_full_random = hyp.get("warmup_full_random", None)
        self.clip_grad_norm = hyp.clip_grad_norm

        self.mse_loss = nn.MSELoss()
        self.params = list(self.policy_model.parameters()) + list(
            self.policy_mixer_model.parameters()
        )
        self.optimizer = torch.optim.RMSprop(
            self.params, lr=self.lr, alpha=0.99, eps=0.00001
        )
        self.epsilon_scheduler = LinearDecayEpsilonScheduler(
            hyp.EPSILON, 0.1, hyp.max_decay_step
        )

    def update(self, *args, **kwargs):
        batch_state, batch_action, batch_reward, _, batch_terminal, batch_other_args = (
            self.get_memory()
        )
        batch_state = torch.tensor(batch_state, dtype=torch.float32).to(self.sync_device)
        batch_action = torch.tensor(batch_action, dtype=torch.long).to(self.sync_device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(self.sync_device)
        batch_terminal = torch.tensor(batch_terminal, dtype=torch.bool).to(self.sync_device)

        (agents_obs,) = batch_other_args
        batch_agents_obs = (
            torch.tensor(agents_obs, dtype=torch.float32).squeeze().to(self.sync_device)
        )

        # cal agent model q
        policy_local_q = self.policy_model(batch_agents_obs)
        target_local_q = self.target_model(batch_agents_obs)

        # mixer model cal policy global q
        policy_choice_local_qs = torch.gather(
            policy_local_q, dim=-1, index=batch_action.unsqueeze(-1)
        ).squeeze()
        policy_global_qs = self.policy_mixer_model(policy_choice_local_qs, batch_state)

        # mixer model cal target global q
        target_max_local_qs = target_local_q.max(dim=-1)[0]
        target_global_qs = self.target_mixer_model(target_max_local_qs, batch_state)

        # cal loss
        target = batch_reward + self.gamma * ~batch_terminal * target_global_qs
        td_error = target.detach() - policy_global_qs
        mean_td_error = td_error.mean()
        loss = (td_error**2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad_norm)
        self.optimizer.step()
        return loss.item(), mean_td_error.item()

    def execute(self, agents_obs):
        agents_obs = torch.FloatTensor(agents_obs).to(self.sync_device)
        agents_Q = self.policy_model(agents_obs)
        return agents_Q

    def save_memory(self, *args):
        assert self.memory is not None
        self.memory.push(*args)

    def get_memory(self):
        assert self.memory is not None
        return self.memory.sample_batch(self.batch_size)

    def sync_update_weights(self):
        self.policy_model.sync_weights_to(self.target_model)
        self.policy_mixer_model.sync_weights_to(self.target_mixer_model)

    @property
    def is_ready(self):
        assert self.memory is not None
        if self.warmup_size is not None:
            assert self.memory.size >= self.warmup_size
        return (
            self.memory.is_overflow
            if self.warmup_size is None
            else len(self.memory) >= self.warmup_size
        )
