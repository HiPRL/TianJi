# -*- coding: utf-8 -*-
import functools
import os
import os.path as osp
from pathlib import Path
from threading import Thread

import torch


def check_file_exist(filepath):
    if not osp.exists(filepath):
        raise FileNotFoundError(f"file '{filepath}' not find")


def check_config_attr(cfg):
    assert hasattr(cfg, "environment"), f'cfg must has "environment" attribute'
    assert hasattr(cfg, "agent"), f'cfg must has "agent" attribute'
    assert hasattr(cfg, "exp"), f'cfg must has "exp" attribute'
    cfg.exp.save_freq = cfg.exp.save_freq if cfg.exp.save_freq > 0 else 1


def increment_path(path, sep="", exist_ok=False, mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )
        for n in range(2, 1024):
            p = f"{path}{sep}{n}{suffix}"
            if not osp.exists(p):
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def make_avg():
    c, a = 0, 0

    def avg(v):
        if v is not None:
            nonlocal c, a
            c += 1
            a += v
            return a / c

    return avg


def thread_async(func):
    @functools.wraps(func)
    def wraper(*args, **kwargs):
        Thread(target=func, args=args, kwargs=kwargs).start()

    return wraper


def function(func, *args, callback=None, **kwargs):
    if callback is None:
        return lambda: func(*args, **kwargs)
    else:
        result = func(*args, **kwargs)
        return lambda: callback(result)


def inspect_entry(func):
    @functools.wraps(func)
    def wraper(x):
        if x:
            return func(x)

    return wraper


def set_threads_resource(core):
    os.environ["OMP_NUM_THREADS"] = str(core)
    os.environ["OPENBLAS_NUM_THREADS"] = str(core)
    os.environ["MKL_NUM_THREADS"] = str(core)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(core)
    os.environ["NUMEXPR_NUM_THREADS"] = str(core)
    torch.set_num_threads(core)


def set_random_seed(env, seed=0):
    seed += 1000000
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.seed(seed)
    print(f"Basic Environmental Information, seed {seed}, {env}.", flush=True)


def init_process(cfg):
    # return value:
    # comm(dict): global、worker、reducer comm info.
    # rank(int): process rank id.
    # info(dict): distributed、is_group base init info.
    try:
        from utils.parallel.distributed import (
            MPIDistributed,
            MPIModelCastWithMasterSlave,
            global_comm,
        )

        parallel_cfg = cfg.parallel_parameters
    except:
        return {
            "comm": {"global_comm": global_comm},
            "rank": 0,
            "info": {"is_distributed": False, "is_group": False},
        }
    else:
        import multiprocessing as mp

        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")

        if global_comm.rank_size > 1:

            def worker_comm_registry(cm, cfg):
                worker_comm_nums = cfg.learner_cfg.num + cfg.actor_cfg.num + 1
                assert cm.rank_size == worker_comm_nums, AttributeError(
                    "rank number of config file does not match actual number of node."
                )

                setattr(cfg.learner_cfg, "actor_num", cfg.actor_cfg.num)
                cm.registry_rank(
                    {
                        "learner": list(range(0, cfg.learner_cfg.num)),
                        "actor": list(range(cfg.learner_cfg.num, worker_comm_nums - 1)),
                        "buffer": [range(worker_comm_nums)[-1]],
                    }
                )

            if hasattr(parallel_cfg, "global_cfg") and parallel_cfg["global_cfg"].get(
                "use_group_parallel", False
            ):

                def reducer_comm_registry(cm, cfg):
                    reducer_comm_nums = cfg.global_cfg.group_num + 1
                    assert cm.rank_size == reducer_comm_nums, AttributeError(
                        "rank number of config file does not match actual number of node."
                    )
                    cm.registry_rank(
                        {
                            "master_learner": range(reducer_comm_nums)[-1],
                            "slave_learner": list(range(reducer_comm_nums)[:-1]),
                        }
                    )

                m_ranks = []
                worker_comm = {}
                reducer_comm = {}
                group_rank_size = (
                    parallel_cfg.learner_cfg.num + parallel_cfg.actor_cfg.num + 1
                )
                for i in range(parallel_cfg.global_cfg.group_num):
                    _comm = global_comm.comm.Create(
                        global_comm.comm.Get_group().Incl(
                            list(range(i * group_rank_size, (i + 1) * group_rank_size))
                        )
                    )
                    worker_comm[i] = _comm
                    m_ranks.append(i * group_rank_size)
                m_ranks.append(parallel_cfg.global_cfg.group_num * group_rank_size)
                reducer_comm[0] = global_comm.comm.Create(
                    global_comm.comm.Get_group().Incl(m_ranks)
                )
                global_comm.convert_comm_cls(
                    worker_comm, MPIDistributed, worker_comm_registry, parallel_cfg
                )
                global_comm.convert_comm_cls(
                    reducer_comm,
                    MPIModelCastWithMasterSlave,
                    reducer_comm_registry,
                    parallel_cfg,
                )
                return {
                    "comm": {
                        "global_comm": global_comm,
                        "worker_comm": worker_comm,
                        "reducer_comm": reducer_comm,
                    },
                    "rank": global_comm.rank,
                    "info": {"is_distributed": True, "is_group": True},
                }
            else:
                worker_comm_registry(global_comm, parallel_cfg)
                return {
                    "comm": {"global_comm": global_comm},
                    "rank": global_comm.rank,
                    "info": {"is_distributed": True, "is_group": False},
                }

        return {
            "comm": {"global_comm": global_comm},
            "rank": 0,
            "info": {"is_distributed": False, "is_group": False},
        }
