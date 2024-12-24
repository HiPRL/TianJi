# -*- coding: utf-8 -*-
import copy

import torch
import torch.nn as nn
from drl.base.embryo import Embryo
from drl.builder import EMBRYOS
from drl.utils.buffers import PrioritizedReplayBuffer
from drl.utils.epsilon_schedulers import (
    ConstantEpsilonScheduler,
    LinearDecayEpsilonScheduler,
)

__all__ = ["DQNPrioritiesHead"]


@EMBRYOS.register_module()
class DQNPrioritiesHead(Embryo):
    def __init__(self, model, hyp):
        super(DQNPrioritiesHead, self).__init__(model)
        self.policy_model = model
        self.target_model = copy.deepcopy(model)

        self.lr = hyp.LR
        self.gamma = hyp.GAMMA
        self.update_target_iter = hyp.TARGET_REPLACE_ITER
        self.batch_size = hyp.batch_size
        self.buffer_size = hyp.buffer_size
        self.warmup_size = hyp.get("warmup_size", None)
        self.warmup_full_random = hyp.get("warmup_full_random", None)

        self.memory = PrioritizedReplayBuffer(self.buffer_size)
        self.hub_loss = nn.HuberLoss(reduction="none")
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)
        self.beta_scheduler = LinearDecayEpsilonScheduler(0.4, 1, 250000)
        self.epsilon_scheduler = LinearDecayEpsilonScheduler(1, hyp.EPSILON, 10000)

    def update(self, *args, **kwargs):
        (update_step,) = args
        (
            batch_state,
            batch_action,
            batch_reward,
            batch_next_state,
            batch_terminal,
            batch_other_args,
        ) = self.get_memory(update_step)
        batch_state = torch.tensor(batch_state, dtype=torch.float).to(self.sync_device)
        batch_action = torch.tensor(batch_action, dtype=torch.long).to(self.sync_device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float).to(self.sync_device)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float).to(
            self.sync_device
        )
        batch_terminal = torch.tensor(batch_terminal, dtype=torch.bool).to(self.sync_device)
        weights, idxes, _ = batch_other_args

        pred_value = self.policy_model(batch_state).gather(1, batch_action)
        with torch.no_grad():
            max_q = self.target_model(batch_next_state).max(1, keepdim=True)[0]
            target_value = batch_reward + ~batch_terminal * self.gamma * max_q
        td_loss = (torch.abs(target_value - pred_value) + 1e-7).data.cpu().numpy()
        self.memory.update_priorities(idxes, td_loss)
        loss = (self.hub_loss(pred_value, target_value) * torch.Tensor(weights)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_model.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

    def execute(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state).squeeze(), 0).to(self.sync_device)
        action_value = self.policy_model(state)
        return action_value

    def save_memory(self, *args):
        assert self.memory is not None
        self.memory.push(*args)

    def get_memory(self, step):
        assert self.memory is not None
        return self.memory.sample_batch(
            self.batch_size, beta=self.beta_scheduler.step(step)
        )

    def sync_update_weights(self):
        self.policy_model.sync_weights_to(self.target_model)

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
