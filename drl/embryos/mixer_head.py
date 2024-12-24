# -*- coding: utf-8 -*-
import copy

import torch
import torch.nn as nn
from drl.base.embryo import Embryo
from drl.builder import EMBRYOS
from drl.utils.epsilon_schedulers import LinearDecayEpsilonScheduler

__all__ = ["MixerHead"]


@EMBRYOS.register_module()
class MixerHead(Embryo):
    def __init__(self, model, mixer_model, hyp):
        super(MixerHead, self).__init__(model)
        self.policy_model = model
        self.target_model = copy.deepcopy(model)

        self.policy_mixer_model = mixer_model
        self.target_mixer_model = copy.deepcopy(mixer_model)

        self.lr = hyp.LR
        self.gamma = hyp.GAMMA
        self.double_q = hyp.double_q
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
            hyp.EPSILON, 0.01, hyp.max_decay_step
        )

    def update(self, *args, **kwargs):
        batch_state, batch_action, batch_reward, _, batch_terminal, batch_other_args = (
            self.get_memory()
        )
        batch_state = torch.tensor(batch_state, dtype=torch.float32).to(self.sync_device)
        batch_action = torch.tensor(batch_action, dtype=torch.long).to(self.sync_device)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(self.sync_device)
        batch_terminal = torch.tensor(batch_terminal, dtype=torch.bool).to(self.sync_device)

        agents_actions, agents_local_obs, vaild_steps = batch_other_args
        n_actions = agents_actions.shape[-1]
        batch_agents_actions = torch.tensor(agents_actions, dtype=torch.long).to(
            self.sync_device
        )
        batch_agents_local_obs = torch.tensor(agents_local_obs, dtype=torch.float32).to(
            self.sync_device
        )
        batch_vaild_steps = torch.tensor(vaild_steps, dtype=torch.long).to(self.sync_device)

        batch_action = batch_action[:, :-1, :]
        batch_reward = batch_reward.unsqueeze(-1)[:, :-1, :]
        batch_terminal = batch_terminal.unsqueeze(-1)[:, :-1, :]
        batch_vaild_steps = batch_vaild_steps.unsqueeze(-1)[:, :-1, :]
        mask = batch_vaild_steps * ~batch_terminal

        # cal agent model q
        policy_local_qs = []
        target_local_qs = []
        for trajectory in range(self.memory.episode_len):
            obs = batch_agents_local_obs[:, trajectory, :, :].reshape(
                -1, batch_agents_local_obs.shape[-1]
            )
            policy_local_q, self.batch_policy_hidden_state = self.policy_model(
                obs, self.batch_policy_hidden_state
            )
            policy_local_q = policy_local_q.reshape(
                self.batch_size, self.policy_mixer_model.agent_num, -1
            )
            policy_local_qs.append(policy_local_q)

            target_local_q, self.batch_target_hidden_state = self.target_model(
                obs, self.batch_target_hidden_state
            )
            target_local_q = target_local_q.view(
                self.batch_size, self.target_mixer_model.agent_num, -1
            )
            target_local_qs.append(target_local_q)

        # mixer model cal policy global q
        policy_local_qs = torch.stack(policy_local_qs, dim=1)
        policy_choice_local_qs = torch.sum(
            policy_local_qs[:, :-1, :, :]
            * nn.functional.one_hot(batch_action, num_classes=n_actions),
            axis=-1,
        )
        policy_global_qs = self.policy_mixer_model(
            policy_choice_local_qs, batch_state[:, :-1, :]
        )

        # mixer model cal target global q
        target_local_qs = torch.stack(target_local_qs[1:], dim=1)
        target_unavailable_actions_mask = (batch_agents_actions[:, 1:, :] == 0).float()
        target_local_qs -= 1e7 * target_unavailable_actions_mask
        if self.double_q:
            local_qs_detach = policy_local_qs.clone().detach()
            unavailable_actions_mask = (batch_agents_actions == 0).to(
                dtype=torch.float32
            )
            local_qs_detach -= 1e7 * unavailable_actions_mask
            cur_max_actions = torch.argmax(
                local_qs_detach[:, 1:], dim=-1, keepdim=False
            )
            cur_max_actions_one_hot = nn.functional.one_hot(
                cur_max_actions, num_classes=n_actions
            ).float()
            target_max_local_qs = torch.sum(
                target_local_qs * cur_max_actions_one_hot, dim=-1
            )
        else:
            target_max_local_qs = target_local_qs.max(dim=3)[0]
        target_global_qs = self.target_mixer_model(
            target_max_local_qs, batch_state[:, 1:, :]
        )

        # cal loss
        target = batch_reward + self.gamma * ~batch_terminal * target_global_qs
        td_error = target.detach() - policy_global_qs
        masked_td_error = td_error * mask
        mean_td_error = masked_td_error.sum() / mask.sum()
        loss = (masked_td_error**2).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad_norm)
        self.optimizer.step()
        return loss.item(), mean_td_error.item()

    def global_grad_norm(self):
        grad_gnorm = 0
        num_none_grads = 0
        for param_group in self.optimizer.param_groups:
            # Make sure we only pass params with grad != None into torch
            # clip_grad_norm_. Would fail otherwise.
            params = list(filter(lambda p: p.grad is not None, param_group["params"]))
            if params:
                # PyTorch clips gradients inplace and returns the norm before clipping
                # We therefore need to compute grad_gnorm further down (fixes #4965)
                global_norm = nn.utils.clip_grad_norm_(params, self.clip_grad_norm)

                if isinstance(global_norm, torch.Tensor):
                    global_norm = global_norm.cpu().numpy()

                grad_gnorm += min(global_norm, self.clip_grad_norm)
            else:
                num_none_grads += 1

        # Note (Kourosh): grads could indeed be zero. This method should still return
        # grad_gnorm in that case.
        if num_none_grads == len(self.optimizer.param_groups):
            # No grads available
            return {}
        return {"grad_gnorm": grad_gnorm}

    def execute(self, agents_obs):
        agents_obs = torch.FloatTensor(agents_obs).to(self.sync_device)
        self.policy_hidden_state = (
            self.policy_hidden_state.unsqueeze(0)
            .expand(1, agents_obs.shape[0], -1)
            .to(self.sync_device)
        )
        agents_Q, self.policy_hidden_state = self.policy_model(
            agents_obs, self.policy_hidden_state
        )
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
