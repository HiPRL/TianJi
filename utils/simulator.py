# -*- coding: utf-8 -*-
from pathlib import Path
from time import sleep

import numpy as np
import torch
from utils.enginer import Enginer

__all__ = ["Simulator"]


class Simulator(Enginer):
    def __init__(self, agent, environment, cfg):
        super(Simulator, self).__init__(agent, environment, cfg)

    def train_episode(self):
        self.call_hook("before_train_episode")
        is_over = False
        self.episode_step = 0
        self.episode_reward = 0
        self.status = self.env.reset()
        while not is_over:
            self.env.render()
            factor = self.call_hook("before_train_step")
            policy_factor, learn_factor = self.reduce_factor(factor)
            if self.agent.alg_type == "off-policy":
                actions = self.agent.policy(*policy_factor)
            elif self.agent.alg_type == "on-policy":
                actions, probs, value = self.agent.policy(*policy_factor)
                learn_factor.insert(0, value)
                learn_factor.insert(0, probs)
            else:
                raise TypeError(
                    "unknown agent algorithm type: %s" % self.agent.alg_type
                )
            self.status_, self.reward, is_over, _ = self.env.step(actions)
            self.call_hook("before_train_learn")
            info = self.agent.learn(
                self._episode,
                self.status,
                actions,
                self.reward,
                self.status_,
                is_over,
                learn_factor,
            )
            self.call_hook("after_train_learn")

            if self.agent.learn_step > 0:
                self.scalar_buffer.update(
                    self.agent.alg_scalar_data, index=self.agent.learn_step
                )
                self.scalar_buffer.update(
                    {"train_step_reward": (self._train_step, np.mean([self.reward]))}
                )

            self.episode_reward += np.mean([self.reward])
            self.status = self.status_
            self._train_step += 1
            self.episode_step += 1
            is_over = (
                True
                if self.episode_flag(self.episode_step)
                else np.array([is_over]).all()
            )
            self.call_hook("after_train_step")
        self._episode += 1
        if self.agent.learn_step > 0:
            self.scalar_buffer.update(
                {
                    "train_episode_step": (self._episode, self.episode_step),
                    "train_episode_reward": (self._episode, self.episode_reward),
                }
            )
        self.call_hook("after_train_episode")
        if hasattr(self, "_logger"):
            if self.agent.learn_step > 0:
                self._logger.info(
                    "learn_step {:<6d} train step {:<7d} train episode: {:<5d} episode_step: {:<8d} episode_reward: {:.8f}".format(
                        self.agent.learn_step,
                        self._train_step,
                        self._episode,
                        self.episode_step,
                        self.episode_reward,
                    )
                )
            else:
                print(
                    "Warmup data collection{}".format("." * (self._episode % 6)),
                    flush=True,
                )
                print("\033[1A\x1b[2K", end="")

    @torch.no_grad()
    def test_episode(
        self,
        is_limit=True,
        is_render=False,
        render_delay=0,
        env_seed=None,
        win_func=None,
    ):
        self.check_eval_env()
        self.eval_env.seed(env_seed)
        wins = []
        episode_test_steps = []
        episode_test_rewards = []
        eval_infos = []
        for _ in range(self.eval_times):
            self.call_hook("before_val_episode")
            is_win = False
            is_over = False
            eval_info = []
            _episode_test_step = 0
            _episode_test_reward = []
            self.eval_status = self.eval_env.reset()
            while not is_over:
                if is_render:
                    self.eval_env.render()
                    if isinstance(render_delay, (int, float)):
                        sleep(render_delay)
                factor = self.call_hook("before_val_step")
                policy_factor, _ = self.reduce_factor(factor)
                if self.agent.alg_type == "off-policy":
                    actions = self.agent.predict(*policy_factor)
                elif self.agent.alg_type == "on-policy":
                    actions, *_ = self.agent.predict(*policy_factor)
                else:
                    raise TypeError(
                        "unknown agent algorithm type: %s" % self.agent.alg_type
                    )
                status_, reward, is_over, _ = self.eval_env.step(actions)
                self.eval_status = status_
                _episode_test_step += 1
                _episode_test_reward.append(np.array([reward]).squeeze().tolist())
                eval_info.append((actions, reward))
                is_over = (
                    True
                    if (is_limit and self.episode_flag(_episode_test_step))
                    else np.array([is_over]).all()
                )
                self.call_hook("after_val_step")
            self.call_hook("after_val_episode")
            if win_func:
                is_win = win_func()
            elif hasattr(self.eval_env, "win_status"):
                is_win = self.eval_env.win_status
            else:
                is_win = self.episode_flag(_episode_test_step)
            if hasattr(self, "_logger"):
                self._logger.info(
                    f"test agent episode step: {_episode_test_step}, agent episode reward: {np.sum(_episode_test_reward)}."
                )
            wins.append(is_win)
            eval_infos.append(eval_info)
            episode_test_steps.append(_episode_test_step)
            episode_test_rewards.append(np.sum(_episode_test_reward))
        win_rate = np.mean(wins)
        self.episode_test_step = np.mean(episode_test_steps)
        self.episode_test_reward = np.mean(episode_test_rewards)
        self.scalar_buffer.update(
            {
                "eval_win_rate": (self._train_step, win_rate),
                "eval_episode_step": (self._train_step, self.episode_test_step),
                "eval_episode_reward": (self._train_step, self.episode_test_reward),
            }
        )
        return win_rate, self.episode_test_step, self.episode_test_reward, eval_infos

    def run(self):
        if self.cfg.resume:
            self.agent.resume(self.cfg.resume)

        self.call_hook("before_run")
        while self.train_flag():
            if self.test_flag():
                self.test_episode()

            self.train_episode()

            if self.save_flag():
                self.agent.save(Path(self.cfg.save_dir) / "models")
        self.env.close()
        self.call_hook("after_run")
