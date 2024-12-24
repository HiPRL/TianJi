# -*- coding: utf-8 -*-
# 返回生产吞吐、消耗吞吐、收敛耗时、收敛步数
# usage: python eval_throughput.py --exp 1dqn_cartpole --batchsize 32

import argparse
import datetime
import os
from glob import glob

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="eval.")
    parser.add_argument(
        "--exp",
        type=str,
        default="experiments/1dqn_cartpole",
        help="training task name.",
    )
    parser.add_argument("--batchsize", type=int, default=32, help="batch size.")
    parser.add_argument("--learnstep", type=int, default=10000, help="all train step.")
    parser.add_argument(
        "--recvsize",
        type=int,
        default=32,
        help="learner recv size, equal of buffer send size.",
    )
    parser.add_argument("--exp-num", type=int, default=5, help="times of experiments.")
    return parser.parse_args()


def product_thrp(txt_paths):
    # 返回生产吞吐
    all_actor_thrp = 0
    for i, path in enumerate(txt_paths):
        with open(path, "r") as fr:
            lines = fr.readlines()
        actor_step = lines[-1].strip().split(",")[3]
        actor_time = lines[-1].strip().split(",")[0]
        actor_thrp = int(actor_step) / float(actor_time)
        print(f"actor {i}, product_thrp {actor_thrp:.2f}")
        all_actor_thrp += actor_thrp
    print(f"all actor, product_thrp {all_actor_thrp:.2f}")
    return all_actor_thrp


def consume_thrp(learner_pths, recvsize=32, batchsize=32, learner_step=10000):
    # 返回learner接收吞吐、消耗吞吐、完成步数、完成耗时
    all_learner_thrp, all_recv_thrp, finish_step, finish_time = 0, 0, 0, 0
    learner_recv_thrp, learner_recv_count, learner_time = 0, 0, 0
    for i, path in enumerate(learner_pths):
        with open(path, "r") as fr:
            lines = fr.readlines()
        for index in range(len(lines) - 1, -1, -1):
            if "cur" in lines[index]:
                learner_step = lines[index].strip().split(" ")[-1]
                break
        for index in range(len(lines) - 1, -1, -1):
            if "work time" in lines[index]:
                learner_time = lines[index].strip().split(" ")[-1][:-1]
                break
        # import pdb; pdb.set_trace()
        learner_thrp = float(learner_step) / float(learner_time)
        print(f"learner {i}, learner_thrp {learner_thrp:.2f}")
        all_learner_thrp += learner_thrp
        finish_time += float(learner_time)
        finish_step += float(learner_step)

        for index in range(len(lines) - 1, -1, -1):
            if "count" in lines[index]:
                learner_recv_count = lines[index].strip().split(" ")[-1]
                break
        learner_recv_thrp = int(learner_recv_count) / float(learner_time)
        print(f"learner {i}, recv_thrp {learner_recv_thrp:.2f}")
        all_recv_thrp += learner_recv_thrp
    finish_time /= len(learner_pths)
    finish_step /= len(learner_pths)
    print(
        f"all learner, learner_recv {all_recv_thrp*recvsize:.2f}, learner_consume_thrp {all_learner_thrp*batchsize:.2f}"
    )

    return (
        all_recv_thrp * recvsize,
        all_learner_thrp * batchsize,
        finish_step,
        finish_time,
    )


def convg(ar_pth, learner_pths):
    # 返回收敛耗时、收敛步数
    with open(ar_pth, "r") as fr:
        lines = fr.readlines()
    convg_time = lines[-4].strip().split(" ")[-1]
    if convg_time == "inf":
        print("convg_time is inf")
        return float("inf"), float("inf")
    else:
        convg_time = float(convg_time)
        print(f"convergance time: {convg_time:.2f}")

        all_convg_steps = 0
        cost_time = int(convg_time)  # 54， 只精确到秒
        for i, path in enumerate(learner_pths):
            with open(path, "r") as fr:
                learner_lines = fr.readlines()
            start_day = learner_lines[0].strip().split(" ")[0]  # '2024-04-28'
            start_hms = (
                learner_lines[0].strip().split(" ")[1].split(",")[0]
            )  # '17:45:48'
            start_time = datetime.datetime.strptime(
                start_day + " " + start_hms, "%Y-%m-%d %H:%M:%S"
            )  # datetime.datetime(2024, 4, 28, 17, 45, 48)
            end_time = start_time + datetime.timedelta(
                seconds=cost_time
            )  # datetime.datetime(2024, 4, 28, 17, 46, 42)
            end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")  # '2024-04-28 17:46:42'

            # 查找end_time，返回对应step
            for line in learner_lines:
                if end_time in line and "cur learner_step" in line:
                    learner_steps = int(line.strip().split(" ")[-1])
                    print(f"learner {i}, convergance step {learner_steps}")
                    all_convg_steps += learner_steps
                    break

        print(f"all learner, convergance step {all_convg_steps}")
        return convg_time, all_convg_steps


if __name__ == "__main__":
    args = parse_args()
    pwd = os.getcwd()
    exp_paths = glob(os.path.join(pwd, args.exp + "?"))
    exp_paths.append(os.path.join(pwd, args.exp))
    print(exp_paths)
    (
        product_thrp_ls,
        consume_thrp_ls,
        learner_recv_ls,
        finish_step_ls,
        finish_time_ls,
        convg_time_ls,
        convg_step_ls,
    ) = ([], [], [], [], [], [], [])
    for i, exp_path in enumerate(exp_paths):
        print("*" * 50)
        print(f"index {i+1} times exp: ")
        txt_paths = glob(os.path.join(exp_path, "Actor*/log/actor_reward.txt"))
        product_thrp_ls.append(product_thrp(txt_paths))

        learner_pts = glob(os.path.join(exp_path, "Learner_comm*/log/*.log"))
        learner_recv, consume_thrp_, finish_step_, finish_time_ = consume_thrp(
            learner_pts,
            recvsize=args.recvsize,
            batchsize=args.batchsize,
            learner_step=args.learnstep,
        )
        learner_recv_ls.append(learner_recv)
        consume_thrp_ls.append(consume_thrp_)
        finish_step_ls.append(finish_step_)
        finish_time_ls.append(finish_time_)

        ar_path = os.path.join(exp_path, "eval_actor.txt")
        if os.path.exists(ar_path):
            convg_time_, all_convg_steps_ = convg(ar_path, learner_pts)
            if convg_time_:
                convg_time_ls.append(convg_time_)
                convg_step_ls.append(all_convg_steps_)
        else:
            print("exec eval_actor first.")
    print("-" * 50)
    print(
        f"{exp_paths[-1]} {len(exp_paths)} times experiments, and the avg result is:\n \
          product_thrp_avg: {np.mean(product_thrp_ls):.2f}\n \
          lrecv_thrp_avg: {np.mean(learner_recv_ls):.2f}\n \
          consume_thrp_avg: {np.mean(consume_thrp_ls):.2f}\n \
          convg_time_avg: {np.mean(convg_time_ls):.2f}\n \
          convg_step_avg: {np.mean(convg_step_ls):.2f}\n \
          fastest convg_time: {np.min(convg_time_ls):.2f}\n \
          fastest convg_step: {np.min(convg_step_ls):.2f}\n \
          slowest convg_time: {np.max(convg_time_ls):.2f}\n \
          slowest convg_step: {np.max(convg_step_ls):.2f}\n \
          finish_step: {np.mean(finish_step_ls):.2f}\n \
          finish_time: {np.mean(finish_time_ls):.2f} "
    )
