# -*- coding: utf-8 -*-
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import deque

from drl.agents import *
from drl.embryos import *
from drl.models import *
from env import build_env
from drl import build_agent
from utils import Config
from utils.simulator import Simulator
from utils.common import check_config_attr



rew_queue = deque(maxlen=100)
reward_step_opt = np.mean
reward_episode_opt = np.sum


def parse_args():
    parser = argparse.ArgumentParser(description="Learner eval.")
    parser.add_argument("--source", type=str, default="./config/dqn/cartpole_config.py", help="experiment param config file")
    parser.add_argument('--exp', nargs='?', const=True, default=False, help='training task name.')
    parser.add_argument("--times", type=int, default=1, help="eval time.")
    return parser.parse_args()

def val_episode(simulator, eval_num=10):
    rws = []
    win_count = 0
    for _ in range(eval_num):
        is_win, step, reward, eval_info = simulator.test_episode()
        _reward = reward_episode_opt(tuple(map(lambda item: reward_step_opt([item]), reward)))
        win_count += is_win
        rws.append(_reward)
        rew_queue.append(_reward)
    return np.mean(rws), rws, win_count / eval_num

def eval_1(simulator, finish_reward=300, eval_time=10):
    rewards = {}
    ret_save_dir = None
    simulator.call_hook('before_run')
    if simulator.cfg.resume:
        if os.path.isdir(simulator.cfg.resume):
            ret_save_dir = simulator.cfg.resume
            pts = glob(os.path.join(simulator.cfg.resume, "*.pt"))
            pts = sorted(pts, key=lambda item: float(re.findall('.*time(.*).pt', item)[0]))
            for i, paths in enumerate(pts):
                cast_time = float(re.findall('.*time(.*).pt', paths)[0])
                simulator.agent.resume(paths)
                val_episode(simulator, eval_num=eval_time)
                if len(rew_queue) == 100:
                    mr = np.mean(rew_queue.copy())
                    rewards[paths] = (mr, cast_time)
                    print("  {:<3d} {:^10.4f}   {:<8.2f} {}".format(i, cast_time, mr, paths), flush=True)
                    if mr >= finish_reward:
                        break
        else:
            ret_save_dir = os.path.dirname(simulator.cfg.resume)
            simulator.agent.resume(simulator.cfg.resume)
            r = val_episode(simulator)
            rewards[simulator.cfg.resume] = r

    simulator.call_hook('after_run')
    simulator.env.close()

    with open(os.path.join(ret_save_dir, "eval_1.txt"), 'w') as fw:
        for key, value in rewards.items():
            fw.write(f"mean reward: {value[0]} cast time: {value[-1]} path: {key}\n")

def eval_2(simulator, finish_reward=300, eval_time=100):
    rewards = {}
    ret_save_dir = None
    simulator.call_hook('before_run')
    if simulator.cfg.resume:
        if os.path.isdir(simulator.cfg.resume):
            ret_save_dir = simulator.cfg.resume
            pts = glob(os.path.join(simulator.cfg.resume, "*.pt"))
            pts = sorted(pts, key=lambda item: float(re.findall('.*time(.*).pt', item)[0]))
            for i, paths in enumerate(pts):
                cast_time = float(re.findall('.*time(.*).pt', paths)[0])
                simulator.agent.resume(paths)
                r = val_episode(simulator, eval_num=eval_time)
                rewards[paths] = r + (cast_time, )
                print("  {:<3d}  {:^10.4f}   {:^8.2f} {}".format(i, cast_time, r[0], paths), flush=True)
                if r[0] >= finish_reward:
                    break
        else:
            ret_save_dir = os.path.dirname(simulator.cfg.resume)
            simulator.agent.resume(simulator.cfg.resume)
            r = val_episode(simulator)
            rewards[simulator.cfg.resume] = r

    simulator.call_hook('after_run')
    simulator.env.close()

    with open(os.path.join(ret_save_dir, "eval_2.txt"), 'w') as fw:
        for key, value in rewards.items():
            fw.write(f"mean reward: {value[0]} cast time: {value[-1]} path: {key}\n")

def single_eval():
    args = parse_args()
    cfg = Config.fromfile(args.source)
    model_path = os.path.join(args.exp, "Learner_0/models")
    cfg.resume = model_path
    cfg.save_dir = None
    check_config_attr(cfg)

    env = build_env(cfg.environment)
    agent = build_agent(cfg.agent)
    simulator = Simulator(agent, env, cfg)
    simulator.register_hook_from_cfg()

    print("the first eval ways, eval time 10 for model in 100 rew queue.")
    print("index   time        reward    path")
    eval_1(simulator, finish_reward=300, eval_time=10)

    print("\nthe second eval ways, eval time 100 for any model.")
    print("index     time       reward    path")
    eval_2(simulator, finish_reward=300, eval_time=100)

def multi_eval():
    ret = {}
    is_draw = False
    args = parse_args()
    cfg = Config.fromfile(args.source)
    model_path = args.exp
    check_config_attr(cfg)

    env = build_env(cfg.environment)
    agent = build_agent(cfg.agent)
    simulator = Simulator(agent, env, cfg)
    simulator.eval_times = args.times
    simulator.register_hook_from_cfg()

    if os.path.isdir(model_path):
        is_draw = True
        ret_save_dir = model_path
        pts = glob(os.path.join(model_path, "*.pt"))
        pts = sorted(pts, key=lambda item: float(re.findall('.*time(.*).pt', item)[0]))
        for i, paths in enumerate(pts):
            name_info = paths.split('_')
            steps = int(name_info[name_info.index('cast') - 1])
            cast_time = float(re.findall('.*time(.*).pt', paths)[0])
            simulator.agent.resume(paths)
            win_rate, episode_test_step, episode_test_reward, _ = simulator.test_episode()
            ret.setdefault("path", []).append(paths)
            ret.setdefault("steps", []).append(steps)
            ret.setdefault("cast_time", []).append(cast_time)
            ret.setdefault("win_rate", []).append(win_rate)
            ret.setdefault("episode_step", []).append(episode_test_step)
            ret.setdefault("episode_reward", []).append(episode_test_reward)
    else:
        ret_save_dir = os.path.dirname(model_path)
        simulator.agent.resume(model_path)
        win_rate, episode_test_step, episode_test_reward, _ = simulator.test_episode()
        print(f"{win_rate} {episode_test_reward} {episode_test_step}")

    if is_draw:
        plt.figure()
        plt.grid(True)
        _, axs = plt.subplots(2, 2)
        p1, p2, p3, p4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

        p1.set_xticks([])
        p1.set_ylabel("mean reward")
        p1.scatter(np.array(ret["cast_time"]), np.array(ret["episode_reward"]), alpha=0.5, marker=".")

        p2.set_xticks([])
        p2.set_yticks([])
        p2.plot(np.array(ret["steps"]), np.array(ret["episode_reward"]), alpha=1, marker=".")

        p3.set_xlabel("time(s)")
        p3.set_ylabel("win rate")
        p3.plot(np.array(ret["cast_time"]), np.array(ret["win_rate"]), alpha=1, marker=".")

        p4.set_yticks([])
        p4.set_xlabel("step")
        p4.plot(np.array(ret["steps"]), np.array(ret["win_rate"]), alpha=0.75, marker=".")

        plt.suptitle("model mean_reward and win_rate curves")
        plt.savefig(os.path.join(ret_save_dir, f"model_{os.path.basename(ret_save_dir)}_eval.png"), dpi=256)
        plt.close()
    
    with open(os.path.join(ret_save_dir, "model_eval.txt"), 'w') as fw:
        for i, item in enumerate(ret["path"]):
            fw.write(f"{item}: {ret['steps'][i]} {ret['cast_time'][i]} {ret['win_rate'][i]} {ret['episode_reward'][i]} {ret['episode_step'][i]}\n")
            print(f"{item}: {ret['steps'][i]} {ret['cast_time'][i]} {ret['win_rate'][i]} {ret['episode_reward'][i]} {ret['episode_step'][i]}")


if __name__ == "__main__":
    multi_eval()