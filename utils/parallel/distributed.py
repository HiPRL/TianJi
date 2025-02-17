# -*- coding: utf-8 -*-
import functools
import time
from typing import Callable

try:
    from yhcomm import DrlMpi, MasterSlaveWithMPI
except:
    raise AttributeError("To use MPI, yhcomm package must be installed.")


def torch_dispath_process_func(role=None):
    """
    role represents a condition, that determine whether the process is running and the role is callable.
    rank get from torch distributed method, but don't use torch distributed and initialized process.
    """

    def call(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if role:
                assert isinstance(role, Callable)
                if role():
                    return func(*args, **kwargs)
            else:
                from torch import distributed as dist

                rank = (
                    dist.get_rank()
                    if dist.is_available() and dist.is_initialized()
                    else 0
                )
                if rank == 0:
                    return func(*args, **kwargs)

        return wrapper

    return call


class MPIDistributed(DrlMpi):
    """
    mpi tag group design:
        buffer tag in 0-9
        leaner tag in 10-9999
        actor tag in 10000-~
    """

    def __init__(self, comm=None, *args, **kwargs):
        super().__init__(comm, *args, **kwargs)
        self.buffer_timeout, self.actor_timeout, self.learner_timeout = 256, 128, 256

    def learner_send(
        self,
        data,
        is_block=True,
        use_buffer=False,
        use_req=False,
        old_req_list=None,
        logger=None,
        use_timeout=True,
    ):
        if use_req:
            if use_timeout:
                for _req in old_req_list:
                    self.learner_flag = False
                    learner_time = time.time()
                    while not self.learner_flag:
                        learner_wait_time = time.time() - learner_time
                        self.learner_flag = _req.test()[0]
                        if learner_wait_time > self.learner_timeout:
                            if logger:
                                logger.info(f"[learner] time out")
                            _req.Cancel()
                            _req = None
                            self.learner_flag = True
            else:
                for _req in old_req_list:
                    _req.wait()

            req = []
            for rank_index in self.actor_rank:
                req.append(
                    self.send(data, rank_index, mpi_tag=10 + rank_index, blocking=False)
                )
            return req
        else:
            for rank_index in self.actor_rank:
                if use_buffer:
                    req = self.comm.Send_init(data, dest=rank_index, tag=10 + rank_index)
                    return req
                else:
                    self.send(
                        data, rank_index, mpi_tag=10 + rank_index, blocking=is_block
                    )

    def actor_recv(self, actor_rank, is_block=True, recv_buffer=None, use_iprobe=False):
        for learn_rank in self.learner_rank:
            if use_iprobe:
                params = None
                recv_flag = self.comm.iprobe(source=learn_rank, tag=10 + actor_rank)
                if recv_flag:
                    params = self.comm.recv(source=learn_rank, tag=10 + actor_rank)
                return params
            elif recv_buffer:
                return self.comm.Recv_init(
                    recv_buffer, source=learn_rank, tag=10 + actor_rank
                )
            elif is_block:
                return self.recv(learn_rank, mpi_tag=10 + actor_rank)
            else:
                return self.ask_recv(learn_rank, mpi_tag=10 + actor_rank)

    def actor_send(
        self,
        data,
        actor_rank,
        is_block=True,
        use_buffer=False,
        use_req=False,
        old_req=None,
        logger=None,
    ):
        if use_req:
            if old_req:
                self.actor_flag = False
                actor_time = time.time()
                while not self.actor_flag:
                    wait_time = time.time() - actor_time
                    self.actor_flag = old_req.test()[0]
                    if wait_time > self.actor_timeout:
                        if logger:
                            logger.info(f"[actor] time out")
                        old_req.Cancel()
                        old_req = None
                        self.actor_flag = True

            req = None
            for buff_rank in self.buffer_rank:
                req = self.send(
                    data, buff_rank, mpi_tag=10000 + actor_rank, blocking=False
                )
            return req
        else:
            for buff_rank in self.buffer_rank:
                if use_buffer:
                    self.comm.Send_init(data, dest=buff_rank, tag=10000 + actor_rank)
                else:
                    self.send(
                        data, buff_rank, mpi_tag=10000 + actor_rank, blocking=is_block
                    )

    def buffer_recv(self, actor_rank, recv_buffer=None, use_iprobe=False):
        if use_iprobe:
            actor_data = None
            recv_flag = self.comm.iprobe(source=actor_rank, tag=10000 + actor_rank)
            if recv_flag:
                actor_data = self.comm.recv(source=actor_rank, tag=10000 + actor_rank)
            return actor_data
        elif recv_buffer:
            return self.comm.Recv_init(
                recv_buffer, source=actor_rank, tag=10000 + actor_rank
            )
        else:
            return self.ask_recv(actor_rank, mpi_tag=10000 + actor_rank)

    def buffer_send(
        self,
        data,
        is_block=True,
        use_buffer=False,
        use_req=False,
        old_req=None,
        logger=None,
        use_timeout=False,
    ):
        if use_req:
            if use_timeout:
                if old_req:
                    self.buffer_flag = False
                    buffer_time = time.time()
                    while not self.buffer_flag:
                        wait_time = time.time() - buffer_time
                        self.buffer_flag = old_req.test()[0]
                        if wait_time > self.buffer_timeout:
                            if logger:
                                logger.info(f"[buffer] time out")
                            old_req.Cancel()
                            old_req = None
                            self.buffer_flag = True

                return self.send(data, self.learner_rank[0], mpi_tag=0, blocking=False)
            else:
                if old_req:
                    old_req.wait()
                return self.send(data, self.learner_rank[0], mpi_tag=0, blocking=False)
        else:
            for learn_rank in self.learner_rank:
                if use_buffer:
                    self.comm.Send_init(data, dest=learn_rank, tag=0)
                else:
                    self.send(data, learn_rank, mpi_tag=0, blocking=is_block)

    def buffer_send_exit(self):
        for learn_rank in self.learner_rank:
            self.send({"exit_flag": True}, learn_rank, mpi_tag=0)

    def learner_recv(self, is_block=True, recv_buffer=None, use_iprobe=False):
        for buff_rank in self.buffer_rank:
            if use_iprobe:
                buff_data = None
                recv_flag = self.comm.iprobe(source=buff_rank, tag=0)
                if recv_flag:
                    buff_data = self.comm.recv(source=buff_rank, tag=0)
                return buff_data
            if recv_buffer:
                return self.comm.Recv_init(recv_buffer, source=buff_rank, tag=0)
            elif is_block:
                return self.recv(buff_rank, mpi_tag=0)
            else:
                return self.ask_recv(buff_rank, mpi_tag=0)

    def learner_send_exit(self):
        for rank_index in self.actor_rank:
            self.send({"exit_flag": True}, rank_index, mpi_tag=10 + rank_index)

    def learner_stop_buffer(self):
        for buffer_rank in self.buffer_rank:
            self.send({"exit_flag": True}, buffer_rank, mpi_tag=4444)

    def actor_stop_buffer(self, actor_rank):
        for buffer_rank in self.buffer_rank:
            self.send({"exit_flag": True}, buffer_rank, mpi_tag=10000 + actor_rank)


class MPIModelCastWithMasterSlave(MasterSlaveWithMPI):
    def __init__(self, comm=None, *args, **kwargs):
        super(MPIModelCastWithMasterSlave, self).__init__(comm, *args, **kwargs)
        self.learner_timeout, self.root_learner_timeout = 256, 256

    def root_send(
        self,
        data,
        old_req_list,
        learner_rank,
        block=False,
        use_timeout=False,
    ):
        if use_timeout:
            for _req in old_req_list:
                self.learner_flag = False
                learner_time = time.time()
                while not self.learner_flag:
                    learner_wait_time = time.time() - learner_time
                    self.learner_flag = _req.test()[0]
                    if learner_wait_time > self.root_learner_timeout:
                        if _req:
                            _req.Cancel()
                        _req = None
                        self.learner_flag = True
        else:
            for _req in old_req_list:
                _req.wait()

        req = []
        for child_rank_index in learner_rank:
            req.append(
                self.send(
                    data,
                    child_rank_index,
                    mpi_tag=1000 + child_rank_index,
                    blocking=block,
                )
            )
        return req

    def root_recv(self, child_rank_index):
        param = None
        recv_flag = self.comm.iprobe(
            source=child_rank_index, tag=100 + child_rank_index
        )

        if recv_flag:
            param = self.comm.recv(source=child_rank_index, tag=100 + child_rank_index)
        return param

    def child_send(self, data, child_rank, old_req_list, use_timeout=False):
        if use_timeout:
            for _req in old_req_list:
                self.learner_flag = False
                learner_time = time.time()
                while not self.learner_flag:
                    learner_wait_time = time.time() - learner_time
                    self.learner_flag = _req.test()[0]
                    if learner_wait_time > self.learner_timeout:
                        if _req:
                            _req.Cancel()
                        _req = None
                        self.learner_flag = True
        else:
            for _req in old_req_list:
                _req.wait()

        req = []
        req.append(
            self.send(
                data, self.root_learner_rank, mpi_tag=100 + child_rank, blocking=False
            )
        )
        return req

    def child_recv(self, child_rank):
        param = None
        recv_flag = self.comm.iprobe(
            source=self.root_learner_rank, tag=1000 + child_rank
        )

        if recv_flag:
            param = self.comm.recv(source=self.root_learner_rank, tag=1000 + child_rank)
        return param


try:
    global_comm = MPIDistributed()
except:
    dispath_process_func = torch_dispath_process_func
else:
    dispath_process_func = global_comm.dispath_process_func
