# -*- coding: utf-8 -*-
import copy

import torch
import torch.nn as nn
from drl.base.embryo import Embryo
from drl.builder import EMBRYOS
from drl.utils.explores import GaussianVibrateExplore

__all__ = ["MADDPGHead"]


@EMBRYOS.register_module()
class MADDPGHead(Embryo):
    def __init__(self, model, hyp):
        super(MADDPGHead, self).__init__(model)
        self.policy_model = model
        self.target_model = copy.deepcopy(model)

        self.actor_lr = hyp.actor_lr
        self.critic_lr = hyp.critic_lr
        self.gamma = hyp.gamma
        self.tau = hyp.tau
        self.update_target_iter = hyp.target_replace_iter
        self.batch_size = hyp.batch_size
        self.buffer_size = hyp.buffer_size
        self.warmup_size = hyp.get("warmup_size", None)
        self.warmup_full_random = hyp.get("warmup_full_random", None)

        self.mse_loss = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(
            self.policy_model.actor.parameters(), lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.policy_model.critic.parameters(), lr=self.critic_lr
        )
        self.explorer = GaussianVibrateExplore()

    def update(self, *data, **kwargs):
        (
            batch_state,
            batch_action,
            batch_reward,
            batch_next_state,
            batch_next_action,
            batch_terminal,
            index,
        ) = data

        rew = torch.tensor(batch_reward, dtype=torch.float).to(self.sync_device)
        done_n = torch.tensor(batch_terminal, dtype=torch.bool).to(self.sync_device)
        obs_n_o = torch.tensor(batch_state, dtype=torch.float).to(self.sync_device)
        obs_n_n = torch.tensor(batch_next_state, dtype=torch.float).to(self.sync_device)
        action_cur_o = torch.tensor(batch_action, dtype=torch.float).to(self.sync_device)
        action_tar = (
            torch.as_tensor(batch_next_action, dtype=torch.float)
            .detach()
            .to(self.sync_device)
        )
        done_n = torch.tensor(batch_terminal, dtype=torch.bool).to(self.sync_device)
        q = self.Q(
            obs_n_o.reshape(obs_n_o.shape[0], -1),
            action_cur_o.reshape(action_cur_o.shape[0], -1),
        ).reshape(
            -1
        )  # tensor
        q_ = self.Q(
            obs_n_n.reshape(obs_n_n.shape[0], -1),
            action_tar.reshape(action_tar.shape[0], -1),
            use_target_model=True,
        ).reshape(
            -1
        )  # tensor
        target_value = q_ * ~done_n.reshape(-1) * self.gamma + rew.reshape(-1)
        critic_loss = self.mse_loss(q, target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_model.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        obs_n_o_index = obs_n_o[:, index, :]
        model_out, policy_c_new = self.policy_model.actor_forward(
            obs_n_o_index.reshape(obs_n_o_index.shape[0], -1), True
        )
        action_cur_o[:, index, :] = policy_c_new
        policy_q = self.Q(
            obs_n_o.reshape(obs_n_o.shape[0], -1),
            action_cur_o.reshape(action_cur_o.shape[0], -1),
        )
        actor_loss = torch.mul(-1, torch.mean(policy_q))
        actor_loss_pse = torch.mean(torch.pow(model_out, 2))
        self.actor_optimizer.zero_grad()
        (1e-3 * actor_loss_pse + actor_loss).backward()
        nn.utils.clip_grad_norm_(self.policy_model.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        return critic_loss.detach().item()

    def execute(self, state, use_target_model=False):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.sync_device)
        action_value = (
            self.target_model.actor_forward(state)
            if use_target_model
            else self.policy_model.actor_forward(state)
        )
        return action_value

    def Q(self, state, act, use_target_model=False):
        q = (
            self.target_model.critic_forward(state.to(self.sync_device), act.to(self.sync_device))
            if use_target_model
            else self.policy_model.critic_forward(
                state.to(self.sync_device), act.to(self.sync_device)
            )
        )
        return q

    def sync_update_weights(self):
        self.policy_model.sync_weights_to(self.target_model, weight=self.tau)
