# -*- coding: utf-8 -*-
import time
import warnings
from collections import OrderedDict, deque, namedtuple
from pathlib import Path

import numpy as np

from drl.utils.buffers import *
from utils.simulator import Simulator

__all__ = ["Actor", "Learner", "LearnerReducer"]


class Role(object):
    def __init__(self, simulator, rank, cfg):
        assert isinstance(simulator, Simulator)
        self._role_rank = rank
        self.cfg = cfg
        self.role = simulator.clone()
        self.runing_flag = True
        self.start_t = time.time()
        self.tirgger = namedtuple("Tirgger", ["tirgger_condition", "tirgger_call"])
        self.tirgger_queue = OrderedDict()
        self.check_hooks()
        self.check_local_buffer()

    @property
    def role_rank(self):
        return self._role_rank

    def task(self):
        raise NotImplementedError()

    def commit_tirgger(self, *tirgger_task):
        if len(tirgger_task) == 3:
            self.tirgger_queue[tirgger_task[-1]] = self.tirgger(*tirgger_task[:-1])
        else:
            self.tirgger_queue[tirgger_task[-1].__name__] = self.tirgger(*tirgger_task)

    def exec_tirgger(self, name=None):
        if name is not None:
            t = self.tirgger_queue[name]
            status = t.tirgger_condition()
            if status:
                t.tirgger_call(status)
        else:
            for t in self.tirgger_queue.values():
                status = t.tirgger_condition()
                if status:
                    t.tirgger_call(status)

    def check_hooks(self):
        self.role.call_hook("before_run")

    def check_local_buffer(self):
        if hasattr(self.role.agent, "memory"):
            setattr(
                self,
                "_".join((self.__class__.__name__.lower(), "buffer")),
                self.role.agent.memory,
            )
        elif hasattr(self.role.agent.embryo, "memory"):
            setattr(
                self,
                "_".join((self.__class__.__name__.lower(), "buffer")),
                self.role.agent.embryo.memory,
            )
        else:
            warnings.warn(f"{self.role.id} is no local buffer in agent.", UserWarning)

    def exit(self):
        self.runing_flag = False

    def __call__(self):
        self.runing_flag = True
        self.task()


class Actor(Role):
    def __init__(self, simulator, rank, cfg, logger):
        super(Actor, self).__init__(simulator, rank, cfg)
        self.logger = logger
        self.actor_global_step = 0
        self.actor_update_times = 0
        self.actor_param_queue = deque(maxlen=3)
        self.actor_log_dir = Path(self.role.save_dir) / "log"
        self.actor_log_dir.mkdir(parents=True, exist_ok=True)
        setattr(self.role, "_logger", logger)

    @property
    def actor_step(self):
        return self.actor_global_step

    @property
    def learn_times(self):
        return self.actor_update_times * self.cfg.learn_size

    def exit(self):
        self.role.call_hook("after_run")
        super().exit()

    def recv_param(self, params):
        self.actor_param_queue.append(params)

    def update_param(self, flag=True, clear_data=True):
        if flag and len(self.actor_param_queue) > 0:
            params = self.actor_param_queue.pop()
            try:
                self.role.agent.update_model_params(params)
                self.actor_update_times += 1
                self.logger.info(
                    f"actor param queue lenght: {len(self.actor_param_queue)}."
                )
                if clear_data:
                    self.actor_buffer.clear()
            except IOError:
                raise IOError(f"update agent param error, got param:\n{params}")
            except Exception as e:
                print(repr(e))

    def sample(self, size):
        return self.actor_buffer.pop(size)

    def task(self):
        explore_records = []
        while self.runing_flag:
            self.role.call_hook("before_train_episode")
            is_over = False
            self.role.episode_step = 0
            self.role.episode_reward = 0
            self.role.status = self.role.env.reset()
            while not is_over:
                self.role.env.render()
                factor = self.role.call_hook("before_train_step")
                policy_factor, learn_factor = self.role.reduce_factor(factor)
                if self.role.agent.alg_type == "off-policy":
                    actions = self.role.agent.policy(
                        *policy_factor, explore_step=self.learn_times
                    )
                elif self.role.agent.alg_type == "on-policy":
                    actions, probs = self.role.agent.policy(
                        *policy_factor, explore_step=self.learn_times
                    )
                    learn_factor.insert(0, probs)
                else:
                    raise IOError(
                        "unknown agent algorithm type: %s" % self.role.agent.alg_type
                    )
                self.role.status_, self.role.reward, done, _ = self.role.env.step(
                    actions
                )
                self.role.call_hook("before_train_learn")
                self.exec_tirgger("recv_params")
                self.actor_buffer.push(
                    self.role._episode,
                    self.role.status,
                    actions,
                    self.role.reward,
                    self.role.status_,
                    done,
                    learn_factor,
                )
                self.exec_tirgger("send_data_to_buffer")
                self.role.call_hook("after_train_learn")
                self.role.scalar_buffer.update(
                    {
                        "actor_samples_buffer": (
                            self.actor_global_step,
                            len(self.actor_buffer),
                        ),
                        "actor_step_reward": (
                            self.actor_global_step,
                            np.mean([self.role.reward]),
                        ),
                    }
                )
                self.role.status = self.role.status_
                self.role.episode_step += 1
                self.actor_global_step += 1
                self.role.episode_reward += np.mean([self.role.reward])
                is_over = (
                    True
                    if self.role.episode_flag(self.role.episode_step)
                    or self.runing_flag == False
                    else np.array([done]).all()
                )
                # Note: immediately updating of the model may affect the uncertainty for this episode sequence smaple.
                self.update_param()
                self.role.call_hook("after_train_step")
            self.role._episode += 1
            self.role.scalar_buffer.update(
                self.role.agent.alg_scalar_data, index=self.role._episode
            )
            self.role.scalar_buffer.update(
                {
                    "actor_episode_step": (self.role._episode, self.role.episode_step),
                    "actor_episode_reward": (
                        self.role._episode,
                        self.role.episode_reward,
                    ),
                }
            )
            self.role.call_hook("after_train_episode")
            self.logger.info(
                "actor episode: {:<4d} episode_step: {:<3d} actor step: {:<5d} episode_reward: {:.7f}".format(
                    self.role._episode,
                    self.role.episode_step,
                    self.actor_global_step,
                    self.role.episode_reward,
                )
            )
            cast_time = float("%.4f" % (time.time() - self.start_t))

            if hasattr(self.role.env, "win_status"):
                is_win = self.role.env.win_status
            else:
                is_win = self.role.episode_flag(self.role.episode_step)

            explore_records.append(
                f"{cast_time}, {self.role.episode_reward}, {int(is_win)}, {self.role.episode_step}, {self.actor_global_step}\n"
            )

        with open(self.actor_log_dir / "actor_reward.txt", "w") as fw:
            for item in explore_records:
                fw.write(item)

        self.role.env.close()
        self.logger.info(f"actor_{self.role_rank} work finish.")


class Learner(Role):
    def __init__(self, simulator, rank, cfg, logger, learn_once_limit=True):
        super(Learner, self).__init__(simulator, rank, cfg)
        self._run_step = 0
        self._digest_times = 0
        self.logger = logger
        self.mean_step = None
        self.val_reward = None
        self.learner_win_ratio = None
        self.learn_once_limit = learn_once_limit
        self.sp_func_list = []
        setattr(self.role, "_logger", logger)

    @property
    def run_step(self):
        return self._run_step

    @property
    def learner_step(self):
        return self.role.agent.learn_step

    def exit(self):
        self.logger.info(
            f"learner work all cast work time: {time.time() - self.start_t}."
        )
        self.role.call_hook("after_run")
        super().exit()

    def get_param(self):
        return self.role.agent.take_model_params()

    def set_param(self, model_params):
        self.role.agent.update_model_params(model_params)

    def digest(self, data):
        self._digest_times += 1
        self.learner_buffer.push(data, force=True)

    def commit_sp_tirgger(self, *args):
        self.commit_tirgger(*args)
        self.sp_func_list.append(args[-1] if len(args) == 3 else args[-1].__name__)

    def exec_sp_tirgger(self):
        for item in self.sp_func_list:
            self.exec_tirgger(item)

    def task(self):
        if self.role.cfg.resume:
            self.role.agent.resume(self.role.cfg.resume)

        learn_once_step = (
            1
            if self.learn_once_limit or self.cfg.actor_num == 1
            else round(np.log(self.cfg.actor_num))
        )
        while self.runing_flag:
            self.exec_tirgger("recv_data_from_buffer")
            if self.role.agent.is_learn():
                # if self.role.test_flag(self.learner_step):
                #     self.eval(reward_opt=self.cfg.eval_val.get("reward_opt"))

                self._run_step += 1
                for _ in range(learn_once_step):
                    self.role.call_hook("before_train_episode")
                    self.role.call_hook("before_train_step")
                    self.role.call_hook("before_train_learn")
                    info = self.role.agent.learn()
                    if self.learner_step > 0:
                        self.role.scalar_buffer.update(
                            self.role.agent.alg_scalar_data, index=self.learner_step
                        )
                    self.role.call_hook("after_train_learn")
                    self.role.call_hook("after_train_step")
                    self.role.call_hook("after_train_episode")
                    self.logger.info(f"cur learner_step: {self.learner_step}")

                    if self.role.save_flag(self.learner_step):
                        cast_time = float("%.4f" % (time.time() - self.start_t))
                        time_str = time.strftime("%Y%m%d_%H_%M_%S")
                        save_path = (
                            Path(self.role.save_dir)
                            / "models"
                            / f"{time_str}_{self.learner_step}_cast_time{cast_time}.pt"
                        )
                        self.role.agent.save(save_path, use_full_path=True)

                self.role.scalar_buffer.update(
                    {
                        "learner_buffer": (self._run_step, len(self.learner_buffer)),
                        "learner_recv_data_times": (self._run_step, self._digest_times),
                    }
                )
                self.exec_tirgger("send_params")
                self.exec_tirgger("exit_manage")
                self.exec_sp_tirgger()
        self.logger.info("learner task finish.")


class LearnerReducer(Role):
    def __init__(self, simulator, rank, cfg, logger):
        super(LearnerReducer, self).__init__(simulator, rank, cfg)
        self._fusion_step = 0
        self._fusion_num = cfg.group_num
        self._stop_step = self.cfg.exit_val.fusion_step
        self.model_list = deque()
        self.fusion_list = deque()
        self.logger = logger
        self.reduce_model = self.role.agent.embryo.model.get_weights()

    @property
    def fusion_step(self):
        return self._fusion_step

    def task(self):
        while self.runing_flag:
            if self._fusion_step > self._stop_step:
                import os; os._exit(-1)
            
            self.exec_tirgger("recv_child_params")
            if len(self.model_list) >= self._fusion_num:
                model_data = [self.model_list.pop() for _ in range(self._fusion_num)]
                for m_k in self.reduce_model.keys():
                    self.reduce_model[m_k] = np.mean(
                        np.stack([m[m_k] for m in model_data]), axis=0
                    )
                self._fusion_step += 1
                self.fusion_list.append(self.reduce_model)
                self.role.agent.update_model_params(self.reduce_model)

                if self.role.save_flag(self.fusion_step * self.cfg.exp.save_freq):
                    cast_time = float("%.4f" % (time.time() - self.start_t))
                    time_str = time.strftime("%Y%m%d_%H_%M_%S")
                    save_path = (
                        Path(self.role.save_dir)
                        / "models"
                        / f"{time_str}_{self.fusion_step}_cast_time{cast_time}.pt"
                    )
                    self.role.agent.save(save_path, use_full_path=True)

            self.exec_tirgger("send_child_params")
            self.exec_tirgger("send_child_exit")
        self.logger.info("reducer fusion model task finish.")
