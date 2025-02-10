# -*- coding: utf-8 -*-
import sys

sys.path.insert(0, "./")
import argparse
from pathlib import Path

from drl.builder import build_agent
from drl.agents import *
from drl.embryos import *
from drl.models import *
from env.builder import build_env
from utils import Config
from utils.common import (
    check_config_attr,
    increment_path,
    init_process,
    set_random_seed,
)
from utils.manager import Manager


def parse_args():
    parser = argparse.ArgumentParser(description="Reinforcement Learning experiments.")
    parser.add_argument(
        "--source",
        type=str,
        default="./config/dqn/cartpole_config.py",
        help="experiment param config file",
    )
    parser.add_argument(
        "--exp-name", type=str, default="cartpole_dqn_exp", help="name of the experiment"
    )
    parser.add_argument(
        "--actor-num", type=int, default=1, help="actor num, e.g 1,2,4,8."
    )
    parser.add_argument(
        "--group-num", type=int, default=1, help="group num, e.g 1,2,4,8."
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="resume agent training simulator task",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="save model in /models directory",
    )
    parser.add_argument(
        "--only-training",
        action="store_true",
        default=False,
        help="only training, no test or val",
    )
    parser.add_argument(
        "--disable-log", action="store_true", default=False, help="disable log print"
    )
    parser.add_argument(
        "--device", default="cpu", help="device used for train, cpu、cuda、mps or mkl"
    )

    return parser.parse_args()


def train(args):
    # 获取参数、参数处理
    source, project_name = args.source, args.exp_name
    cfg = Config.fromfile(source)
    cfg.project_name = "exp" if project_name is None else project_name
    cfg.save_dir = increment_path(Path("./experiments") / project_name)
    cfg.resume = args.resume
    cfg.is_save_model = args.save_model
    cfg.only_training = args.only_training
    cfg.log_status = args.disable_log
    cfg.parallel_parameters.actor_cfg.num = args.actor_num
    cfg.parallel_parameters.global_cfg.group_num = args.group_num
    cfg.device = args.device
    check_config_attr(cfg)

    # 进程初始化
    comm = init_process(cfg)

    # 环境初始化
    env = build_env(cfg.environment)
    set_random_seed(env, seed=comm["rank"])

    # 搭建智能体
    agent = build_agent(cfg.agent, cfg.device)

    # 训练
    manager = Manager(agent, env, cfg, comm)
    manager.run()


if __name__ == "__main__":
    opt = parse_args()
    train(opt)
