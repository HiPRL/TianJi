# -*- coding: utf-8 -*-
import sys
from copy import deepcopy

from drl.utils.buffers import build_buffer
from utils import make_logger
from utils.common import set_threads_resource
from utils.parallel.worker import Actor, Learner, LearnerReducer
from utils.simulator import Simulator

__all__ = ["Manager"]


class LearnerWork:
    def __init__(self, comm, simulator, rank, cfg, logger):
        super(LearnerWork, self).__init__()
        self.learner = Learner(simulator, rank, cfg, logger)
        self.learner_comm = comm
        self.root_comm = None
        self.cfg = cfg
        self.logger = logger
        self.grand_total = 0
        self.send_params_req = []
        self.send_root_params_req = []

    def run(self):
        self.connect()
        self.learner.commit_tirgger(
            lambda: (
                hasattr(self.cfg.exit_val, "learn_step")
                and self.learner.learner_step
                and self.learner.learner_step > self.cfg.exit_val.learn_step
            )
            or (
                hasattr(self.cfg.exit_val, "reward")
                and self.learner.val_reward
                and self.learner.val_reward > self.cfg.exit_val.reward
            )
            or (
                hasattr(self.cfg.exit_val, "win_ratio")
                and self.learner.learner_win_ratio
                and self.learner.learner_win_ratio > self.cfg.exit_val.win_ratio
            ),
            self.exit_manage,
        )
        self.learner.commit_tirgger(
            lambda: self.learner.run_step % self.cfg.send_interval == 0,
            self.send_params,
        )
        self.learner.commit_tirgger(
            self.recv_data_condition, self.recv_data_from_buffer
        )
        self.learner()

    def connect(self):
        while True:
            for actor_rank in self.learner_comm.actor_rank:
                res_flag = self.learner_comm.comm.sendrecv(0, dest=actor_rank)
                if res_flag:
                    self.logger.info(f"the learner connection with rank {actor_rank}.")
                else:
                    self.logger.info(
                        f"the learner connection failed with rank {actor_rank}."
                    )
            break

    def recv_data_condition(self):
        try:
            return self.learner_comm.learner_recv(use_iprobe=True)
        except Exception as e:
            self.logger.warning(f"learner_recv failed, {e}")

    def recv_data_from_buffer(self, data):
        try:
            if data:
                self.learner.digest(data)
                self.grand_total += 1
                self.logger.info(
                    f"learner recv data({sys.getsizeof(data)} bytes) from buffer success, learner buffer len: {len(self.learner.learner_buffer)}, learner buffer is ready: {self.learner.role.agent.is_learn()}."
                )
        except Exception as e:
            self.logger.warning(f"learner recv data from buffer failed. {e}")
        finally:
            data = None

    def send_params(self, _=None):
        params = deepcopy(self.learner.get_param())
        self.send_params_req = self.learner_comm.learner_send(
            params,
            use_req=True,
            old_req_list=self.send_params_req,
            logger=self.logger,
            use_timeout=True,
        )
        self.logger.info(
            f"learner send params success, learner_step: {self.learner.learner_step}"
        )

    def recv_root_params_condition_force_block(self):
        assert self.root_comm is not None
        try:
            recv_root_params_flag = False
            while not recv_root_params_flag:
                model_params = self.root_comm.child_recv(self.root_comm.rank)
                if model_params:
                    return model_params

        except Exception as e:
            self.logger.warning(f"learner recv root params failed, {e}.")

    def recv_root_params_condition(self):
        assert self.root_comm is not None
        try:
            model_params = self.root_comm.child_recv(self.root_comm.rank)
            if model_params:
                if hasattr(model_params, "get") and model_params.get(
                    "exit_flag", False
                ):
                    self.exit_fmroot()
                else:
                    return model_params
        except Exception as e:
            self.logger.warning(f"learner recv root params failed, {e}.")

    def sync_params(self, model_params, sync_actor_force=True):
        self.learner.set_param(model_params)
        self.logger.info(f"child learner async params success.")
        if sync_actor_force:
            self.send_params_req = self.learner_comm.learner_send(
                model_params,
                use_req=True,
                old_req_list=self.send_params_req,
                use_timeout=True
            )
            self.logger.info(
                f"child learner async actor params success, learner_step: {self.learner.learner_step}"
            )

    def send_root_params(self, _=None):
        assert self.root_comm is not None
        params = deepcopy(self.learner.get_param())
        self.send_root_params_req = self.root_comm.child_send(
            params,
            self.root_comm.rank,
            self.send_root_params_req,
            use_timeout=True
        )
        self.logger.info(f"learner send params to root success.")

    def exit_manage(self, _=None):
        """
        There are two variables used for judgment, grand_total or learner_step.
        grand_total mean number of times received data from buffer.
        """
        import os; os._exit(-1)
        self.learner_comm.learner_send_exit()

        while True:
            ret = self.learner_comm.learner_recv(use_iprobe=True)
            if ret:
                if hasattr(ret, "get") and ret.get("exit_flag", False):
                    break
        if self.root_comm:
            self.send_root_params_req = self.root_comm.child_send(
                {"exit_flag": True}, self.root_comm.rank, self.send_root_params_req
            )
        self.learner.exit()

    def exit_fmroot(self):
        """
        recv exit flag from root-learner, and send exit-flag to actors, and exit itself.
        """
        self.learner_comm.learner_send_exit()

        while True:
            ret = self.learner_comm.learner_recv(use_iprobe=True)
            if ret:
                if hasattr(ret, "get") and ret.get("exit_flag", False):
                    break
        self.send_root_params_req = self.root_comm.child_send(
            {"exit_flag": True}, self.root_comm.rank, self.send_root_params_req
        )
        self.learner.exit()


class ActorWork:
    def __init__(self, comm, simulator, rank, cfg, logger):
        super(ActorWork, self).__init__()
        self.actor = Actor(simulator, rank, cfg, logger)
        self.actor_comm = comm
        self.actor_rank = rank
        self.cfg = cfg
        self.logger = logger
        self.send_data_req = None
        self.data_send_flag = False

    def run(self):
        self.connect()
        if self.actor.role.agent.__class__.__name__.lower()  == "ppo":
            self.actor.commit_tirgger(self.recv_params_condition_force_block, self.recv_params)
        else:
            self.actor.commit_tirgger(self.recv_params_condition, self.recv_params)
        self.actor.commit_tirgger(
            lambda: len(self.actor.actor_buffer) >= self.cfg.send_size,
            self.send_data_to_buffer,
        )
        self.actor()

    def connect(self):
        while True:
            res_flag = self.actor_comm.comm.recv(source=0)
            self.actor_comm.comm.send(res_flag + 1, dest=0)
            break

    def send_data_to_buffer(self, send_size=None):
        try:
            data = self.actor.sample(self.cfg.send_size)
            self.send_data_req = self.actor_comm.actor_send(
                data,
                self.actor.role_rank,
                use_req=True,
                old_req=self.send_data_req,
                logger=self.logger,
            )
            self.data_send_flag = True
        except Exception as e:
            self.logger.warning(f"actor send data to buffer failed. {e}")
        else:
            self.logger.info(f"actor send data to buffer success, data: {len(data)}.")
        finally:
            data = None

    def recv_params_condition(self):
        try:
            model_params = self.actor_comm.actor_recv(
                self.actor.role_rank, use_iprobe=True
            )
            if model_params:
                return model_params
        except Exception as e:
            self.logger.warning(f"actor_recv failed, {e}.")

    def recv_params_condition_force_block(self):
        try:
            model_params = self.actor_comm.actor_recv(
                self.actor.role_rank, use_iprobe=True
            )

            while self.data_send_flag:
                model_params = self.actor_comm.actor_recv(
                    self.actor.role_rank, use_iprobe=True
                )
                if model_params:
                    self.data_send_flag = False
                    return model_params
        except Exception as e:
            self.logger.warning(f"actor_recv failed, {e}.")

    def recv_params(self, model_params=None):
        try:
            if model_params:
                if hasattr(model_params, "get") and model_params.get(
                    "exit_flag", False
                ):
                    self.actor.exit()
                    self.actor_comm.actor_stop_buffer(self.actor.role_rank)
                    self.logger.info(
                        f"actor_{self.actor.role_rank} exit listen task finish."
                    )
                else:
                    self.actor.recv_param(model_params)
                    self.logger.info(
                        f"actor_{self.actor.role_rank} recv params({sys.getsizeof(model_params)} bytes) success."
                    )
        except Exception as e:
            self.logger.warning(f"actor recv params failed, {e}.")


class BufferWork:
    def __init__(self, comm, rank, cfg, logger):
        super(BufferWork, self).__init__()
        self.buff = build_buffer(cfg.global_buffer)
        self.buff_comm = comm
        self.buff_rank = rank
        self.cfg = cfg
        self.logger = logger
        self.runing_flag = True
        self.send_data_req = None
        self.actor_send_exit_flag = 0

    def run(self):
        while self.runing_flag:
            for rank_index in self.buff_comm.actor_rank:
                try:
                    actor_data = self.buff_comm.buffer_recv(rank_index, use_iprobe=True)
                    if actor_data:
                        if hasattr(actor_data, "get") and actor_data.get(
                            "exit_flag", False
                        ):
                            self.actor_send_exit_flag += 1
                            self.logger.info(
                                f"buffer listen actor_{rank_index} rank exit flag, {self.actor_send_exit_flag} actor task finish."
                            )
                            if self.actor_send_exit_flag == len(
                                self.buff_comm.actor_rank
                            ):
                                self.runing_flag = False
                            break
                        self.buff.push(actor_data, force=True)
                        self.logger.info(
                            f"buffer recv data({sys.getsizeof(actor_data)} bytes) success, buffer len: {len(self.buff)}."
                        )
                except Exception as e:
                    self.logger.warning(f"buffer recv data failed. {e}")
                finally:
                    actor_data = None

                if len(self.buff) >= self.cfg.send_size:
                    try:
                        buff_data = self.buff.pop(self.cfg.send_size)
                    except Exception as e:
                        self.logger.warning(f"buffer send data failed. {e}")
                    else:
                        self.send_data_req = self.buff_comm.buffer_send(
                            buff_data,
                            use_req=True,
                            old_req=self.send_data_req,
                            logger=self.logger,
                        )
                        self.logger.info(
                            f"buffer send data success, data: {len(buff_data)}."
                        )
                    finally:
                        buff_data = None

        self.buff_comm.buffer_send(
            {"exit_flag": True},
            use_req=True,
            old_req=self.send_data_req,
            use_timeout=True,
            logger=self.logger,
        )
        self.logger.info("buffer task finish.")

    @property
    def rank_id(self):
        return self.buff_rank


class LearnerReducerWork:
    def __init__(self, comm, simulator, rank, cfg, logger):
        super(LearnerReducerWork, self).__init__()
        self.reducer = LearnerReducer(simulator, rank, cfg, logger)
        self.reducer_comm = comm
        self.cfg = cfg
        self.logger = logger
        self.send_child_params_req = []
        self.recv_child_exit_flag = 0

    def run(self):
        self.reducer.commit_tirgger(
            self.recv_child_params_condition, self.recv_child_params
        )
        self.reducer.commit_tirgger(
            lambda: len(self.reducer.fusion_list) > 0, self.send_child_params
        )
        self.reducer.commit_tirgger(
            lambda: (
                hasattr(self.cfg, "fusion_step")
                and self.reducer.fusion_step > self.cfg.fusion_step
            ),
            self.send_child_exit,
        )
        self.reducer()

    def recv_child_params_condition(self):
        return len(self.reducer.model_list) < self.reducer._fusion_num

    def recv_child_params(self, _=None):
        try:
            for child_rank in self.reducer_comm.child_learner_rank:
                param = self.reducer_comm.root_recv(child_rank)
                if param:
                    if hasattr(param, "get") and param.get("exit_flag", False):
                        self.recv_child_exit_flag += 1
                        self.logger.info(
                            f"reducer listen child learner {child_rank} rank exit flag, {self.recv_child_exit_flag} child learner task finish."
                        )
                        if self.recv_child_exit_flag == len(
                            self.reducer_comm.child_learner_rank
                        ):
                            self.reducer.runing_flag = False
                    else:
                        self.reducer.model_list.append(param)
                        self.logger.info(
                            f"reducer recv learner_{child_rank} param success, reducer model_list len: {len(self.reducer.model_list)}."
                        )
        except Exception as e:
            self.logger.warning(f"reducer recv param failed. {e}")

    def send_child_exit(self, _=None):
        self.logger.info(
            f"root fusion_step {self.reducer.fusion_step}, send exit flag to child"
        )
        self.send_child_params_req = self.reducer_comm.root_send(
            {"exit_flag": True},
            self.send_child_params_req,
            self.reducer_comm.child_learner_rank,
            block=True,
        )
        self.logger.info(
            f"reducer send params success, reducer_step: {self.reducer._fusion_step}"
        )
        self.reducer.runing_flag = False

    def send_child_params(self, _=None):
        params = self.reducer.fusion_list.pop()
        self.send_child_params_req = self.reducer_comm.root_send(
            params,
            self.send_child_params_req,
            self.reducer_comm.child_learner_rank,
        )
        self.logger.info(
            f"reducer send params success, reducer_step: {self.reducer._fusion_step}"
        )


class Manager(object):
    def __init__(self, agent, env, cfg, comm):
        self.simulator = Simulator(agent, env, cfg)
        self.cfg = cfg
        self.comm = comm
        self.is_group = comm["info"]["is_group"]
        self.is_distributed = comm["info"]["is_distributed"]
        self.parallel_cfg = cfg.parallel_parameters if self.is_distributed else None
        self.simulator.register_hook_from_cfg()

    def run(self):
        print(
            f"Start of intelligent agent training, agent obj info {self.simulator.agent}.",
            flush=True,
        )
        if self.is_distributed:
            if self.is_group:
                self.group_run()
            else:
                ret = self.parallel_run(self.comm["comm"]["global_comm"])
                if ret:
                    ret.run()
        else:
            self.direct_run()

    def direct_run(self):
        logger = make_logger(
            self.cfg.project_name,
            self.cfg.save_dir / "log" / (self.cfg.project_name + "_log.log"),
            disable=self.cfg.log_status,
        )
        setattr(self.simulator, "_logger", logger)
        self.simulator.run()

    def parallel_run(self, comm):
        if comm.is_learner:
            save_dir = self.cfg.save_dir / f"Learner_{comm.name}_{comm.rank}"
            self.simulator.save_dir = save_dir
            logger = make_logger(
                f"Learner_{comm.name}_{comm.rank}",
                save_dir / "log" / (f"Learner_{comm.name}_{comm.rank}" + "_log.log"),
                disable=self.cfg.log_status,
            )
            if hasattr(self.parallel_cfg.learner_cfg, "cores"):
                set_threads_resource(self.parallel_cfg.learner_cfg.cores)
            return LearnerWork(
                comm, self.simulator, comm.rank, self.parallel_cfg.learner_cfg, logger
            )
        elif comm.is_actor:
            save_dir = self.cfg.save_dir / f"Actor_{comm.name}_{comm.rank}"
            self.simulator.save_dir = save_dir
            #self.simulator.agent.set_device('cpu')
            self.parallel_cfg.actor_cfg.learn_size = (
                self.parallel_cfg.learner_cfg.send_interval
            )
            logger = make_logger(
                f"Actor_{comm.name}_{comm.rank}",
                save_dir / "log" / (f"Actor_{comm.name}_{comm.rank}" + "_log.log"),
                disable=self.cfg.log_status,
            )
            if hasattr(self.parallel_cfg.actor_cfg, "cores"):
                set_threads_resource(self.parallel_cfg.actor_cfg.cores)
            return ActorWork(
                comm, self.simulator, comm.rank, self.parallel_cfg.actor_cfg, logger
            )
        elif comm.is_buffer:
            save_dir = self.cfg.save_dir / f"Buffer_{comm.name}_{comm.rank}"
            self.parallel_cfg.buffer_cfg.send_size = (
                self.parallel_cfg.buffer_cfg.send_size * self.parallel_cfg.actor_cfg.num
            )
            logger = make_logger(
                f"Buffer_{comm.name}_{comm.rank}",
                save_dir / "log" / (f"Buffer_{comm.name}_{comm.rank}" + "_log.log"),
                disable=self.cfg.log_status,
            )
            del self.simulator
            if hasattr(self.parallel_cfg.buffer_cfg, "cores"):
                set_threads_resource(self.parallel_cfg.buffer_cfg.cores)
            return BufferWork(comm, comm.rank, self.parallel_cfg.buffer_cfg, logger)
        else:
            print(f"the rank {comm.rank} nothing to do.", flush=True)

    def group_run(self):
        global_comm = self.comm["comm"]["global_comm"]
        work_comm = self.comm["comm"]["worker_comm"]
        reduce_comm = self.comm["comm"]["reducer_comm"]

        for cm in work_comm.values():
            if global_comm.distinguish_comm(cm):
                ret = self.parallel_run(cm)
                if ret:
                    if isinstance(ret, LearnerWork):
                        setattr(ret, "root_comm", reduce_comm[0])
                        ret.learner.commit_sp_tirgger(
                            lambda: ret.learner.run_step % ret.cfg.send_root_interval
                            == 0,
                            ret.send_root_params,
                        )
                        if ret.learner.role.agent.__class__.__name__.lower() == "ppo":
                            ret.learner.commit_sp_tirgger(
                                ret.recv_root_params_condition_force_block, ret.sync_params
                            )
                        else:
                            ret.learner.commit_sp_tirgger(
                                ret.recv_root_params_condition, ret.sync_params
                            )
                    ret.run()

        for cm in reduce_comm.values():
            if global_comm.distinguish_comm(cm) and cm.is_master_learner:
                save_dir = self.cfg.save_dir / f"Learner_Manager"
                self.simulator.save_dir = save_dir
                logger = make_logger(
                    f"LearnerManager",
                    save_dir / "log" / (f"Learner_Manager" + "_log.log"),
                    disable=self.cfg.log_status,
                )
                LearnerReducerWork(
                    cm, self.simulator, cm.rank, self.parallel_cfg.global_cfg, logger
                ).run()
