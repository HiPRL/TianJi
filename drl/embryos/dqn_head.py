# -*- coding: utf-8 -*-
import copy

import torch
import torch.nn as nn
from drl.base.embryo import Embryo
from drl.builder import EMBRYOS
from drl.utils.buffers import StepBuffer
from drl.utils.epsilon_schedulers import (
    ConstantEpsilonScheduler,
    LinearDecayEpsilonScheduler,
)

__all__ = ["DQNHead"]


@EMBRYOS.register_module()
class DQNHead(Embryo):
    def __init__(self, model, hyp):
        super(DQNHead, self).__init__(model)
        self.policy_model = model
        self.target_model = copy.deepcopy(model)

        self.lr = hyp.LR
        self.gamma = hyp.GAMMA
        self.update_target_iter = hyp.TARGET_REPLACE_ITER
        self.batch_size = hyp.batch_size
        self.buffer_size = hyp.buffer_size
        self.warmup_size = hyp.get("warmup_size", None)
        self.warmup_full_random = hyp.get("warmup_full_random", None)

        self.memory = StepBuffer(self.buffer_size)
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=self.lr)
        self.epsilon_scheduler = LinearDecayEpsilonScheduler(1, hyp.EPSILON, 10000)

    def update(self, *args, **kwargs):
        batch_state, batch_action, batch_reward, batch_next_state, batch_terminal, _ = (
            self.get_memory()
        )
        batch_state = torch.tensor(batch_state, dtype=torch.float).to(self.sync_device)
        batch_action = torch.tensor(batch_action, dtype=torch.long).to(self.sync_device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float).to(self.sync_device)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float).to(
            self.sync_device
        )
        batch_terminal = torch.tensor(batch_terminal, dtype=torch.bool).to(self.sync_device)

        pred_value = self.policy_model(batch_state).gather(1, batch_action)
        with torch.no_grad():
            max_q = self.target_model(batch_next_state).max(1, keepdim=True)[0]
            target_value = batch_reward + ~batch_terminal * self.gamma * max_q
        loss = self.mse_loss(pred_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def execute(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state).squeeze(), 0).to(self.sync_device)
        action_value = self.policy_model(state)
        return action_value

    def save_memory(self, *args):
        assert self.memory is not None
        self.memory.push(*args)

    def get_memory(self):
        assert self.memory is not None
        return self.memory.sample_batch(self.batch_size)

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
