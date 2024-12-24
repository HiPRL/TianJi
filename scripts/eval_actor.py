# -*- coding: utf-8 -*-
import argparse
import os
from collections import OrderedDict
from glob import glob
from typing import List, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["axes.unicode_minus"] = False
color_table = list(mcolors.CSS4_COLORS.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Actor eval.")
    parser.add_argument(
        "--exp", nargs="?", const=True, default=False, help="training task dir."
    )
    parser.add_argument(
        "--exp-name", nargs="?", const=True, default=False, help="training task name."
    )
    parser.add_argument(
        "--finish-reward", type=float, default=None, help="agent target reward."
    )
    parser.add_argument(
        "--mode",
        choices=["once", "multi_time", "multi_episode", "multi_actor"],
        default="once",
        help="The eval mode.",
    )
    parser.add_argument("--exp-prefix", default="", help="exp prefix, e.g 1,2,4,8.")
    parser.add_argument(
        "--exp-suffix",
        default="",
        help="exp suffix, e.g 5,5,5,5 or [1,3],(0,2),[2,3,4],5.",
    )
    parser.add_argument(
        "--eval-func", type=int, default=0, help="multi exp eval function."
    )
    return parser.parse_args()


def read_lines(txt_path):
    with open(txt_path, "r") as fr:
        return fr.readlines()


def eval_once(exp_path, finish_reward=None, is_draw=True):
    # 一个实验
    info = {}
    reward_list = []
    txt_paths = glob(os.path.join(exp_path, "Actor*/*/actor_reward.txt"))
    save_path = os.path.join(exp_path, "eval_actor.txt")
    for path in txt_paths:
        for item in read_lines(path):
            content = item.strip().split(",")
            if content:
                reward_list.append(
                    (float(content[0]), float(content[1]), int(content[2]))
                )

    finish_time, finish_flag = np.inf, True
    ep_times, ep_rewards, ep_wins = [], [], []
    fw = open(save_path, "w", encoding="utf-8")
    win_fw = open(os.path.join(exp_path, "actor_win_data.txt"), "w", encoding="utf-8")
    sort_list = sorted(reward_list, key=lambda item: item[0])
    for i in range(100, len(sort_list)):
        datas = sort_list[i - 100 : i]
        x = np.mean(list(map(lambda item: item[1], datas)))
        win_rate = np.nanmean(list(map(lambda item: item[2], datas)))
        fw.write(f"{i} {sort_list[i][0]} {x}\n")
        win_fw.write(f"{i} {sort_list[i][0]} {win_rate}\n")
        ep_rewards.append(x)
        ep_wins.append(win_rate)
        ep_times.append(sort_list[i][0])
        if finish_reward and finish_flag and x >= finish_reward:
            finish_flag = False
            finish_time = sort_list[i][0]
    info["end_time"] = ep_times[-1]
    info["win_rate"] = ep_wins[-1]
    info["finish_time"] = finish_time
    info["min_reward"] = np.min(ep_rewards)
    info["max_reward"] = np.max(ep_rewards)
    info["mean_rewaed"] = np.mean(ep_rewards)
    fw.write("\nactor reward summary:\n")
    for key, value in info.items():
        fw.write(f"{key}: {value}\n")
    fw.close()
    win_fw.close()

    if is_draw:
        plt.figure()
        plt.grid(True)
        plt.title("actor mean reward curves")
        plt.xlabel("time(s)")
        plt.ylabel("mean reward")
        plt.scatter(np.array(ep_times), np.array(ep_rewards), alpha=0.5, marker=".")
        plt.savefig(
            os.path.join(exp_path, f"{os.path.basename(exp_path)}_reward.png"), dpi=256
        )
        plt.close()

    return ep_times, ep_rewards, ep_wins, info


def eval_time_multi(
    exp_dir, exp_name, finish_reward=None, is_draw=True, exp_indexs=5, range_slice=256
):
    # 一组实验，时间：一个实验重复多次，在于后缀
    # exp_indexs是int，默认为range(exp_indexs)
    # exp_indexs是list、tuple，指定实验名称后缀，0和1都表示无后缀
    if isinstance(exp_indexs, int):
        exp_names = [exp_name]
        for i in range(2, exp_indexs + 1):
            exp_names.append(f"{exp_name}{i}")
    elif isinstance(exp_indexs, (tuple, list)):
        exp_names = []
        for index in exp_indexs:
            if isinstance(index, list):
                for item in index:
                    if item == 0 or item == 1:
                        exp_names.append(exp_name)
                    else:
                        exp_names.append(f"{exp_name}{item}")
            else:
                if index == 0 or index == 1:
                    exp_names.append(exp_name)
                else:
                    exp_names.append(f"{exp_name}{index}")
        exp_names = OrderedDict.fromkeys(exp_names).keys()
    else:
        raise TypeError(
            f"exp_indexs type need int、list、tuple， but got {type(exp_indexs)}"
        )

    eval_multi_info = {}
    multi_ep_times, multi_ep_rewards, multi_info = [], [], []
    for name in exp_names:
        ep_times, ep_rewards, ep_wins, info = eval_once(
            os.path.join(exp_dir, name), finish_reward=finish_reward, is_draw=is_draw
        )
        multi_ep_times.append(ep_times)
        multi_ep_rewards.append(ep_rewards)
        multi_info.append(info)
    align_time = np.min([item["end_time"] for item in multi_info])
    range_time = np.linspace(0, align_time, range_slice)
    merge_data = []
    for _ep_times, _ep_rewards in zip(multi_ep_times, multi_ep_rewards):
        for t, r in zip(_ep_times, _ep_rewards):
            merge_data.append((t, r))
    merge_data = sorted(merge_data, key=lambda item: item[0])

    align_finish_max_time, finish_max_flag, align_finish_max_range_reward = (
        np.inf,
        True,
        [],
    )
    align_finish_mean_time, finish_mean_flag, align_finish_mean_range_reward = (
        np.inf,
        True,
        [],
    )
    align_min_reward, align_mean_reward, align_max_reward = (
        [merge_data[0][1]],
        [merge_data[0][1]],
        [merge_data[0][1]],
    )
    for i, item in enumerate(range_time[1:]):
        range_min_time = range_time[i - 1]
        range_max_time = item
        range_data = list(
            filter(lambda t: range_min_time < t[0] <= range_max_time, merge_data)
        )
        range_data = [item[1] for item in range_data]
        if len(range_data) == 0:
            align_min_reward.append(align_min_reward[-1])
            align_mean_reward.append(align_mean_reward[-1])
            align_max_reward.append(align_max_reward[-1])
            continue
        _max_reward = np.max(range_data)
        _mean_reward = np.mean(range_data)
        align_min_reward.append(np.min(range_data))
        align_max_reward.append(_max_reward)
        align_mean_reward.append(_mean_reward)
        if finish_reward and finish_max_flag and _max_reward >= finish_reward:
            finish_max_flag = False
            align_finish_max_time = item
            align_finish_max_range_reward = range_data

        if finish_reward and finish_mean_flag and _mean_reward >= finish_reward:
            finish_mean_flag = False
            align_finish_mean_time = item
            align_finish_mean_range_reward = range_data

    eval_multi_info["all_reward_length"] = len(merge_data)
    eval_multi_info["all_min_reward"] = np.min(
        [item["min_reward"] for item in multi_info]
    )
    eval_multi_info["all_max_reward"] = np.max(
        [item["max_reward"] for item in multi_info]
    )
    eval_multi_info["min_run_time"] = np.min([item["end_time"] for item in multi_info])
    eval_multi_info["max_run_time"] = np.max([item["end_time"] for item in multi_info])
    eval_multi_info["fast_finish_time"] = np.min(
        [item["finish_time"] for item in multi_info]
    )
    eval_multi_info["align_time"] = align_time
    eval_multi_info["align_finish_max_time"] = align_finish_max_time
    eval_multi_info["align_finish_mean_time"] = align_finish_mean_time
    eval_multi_info["align_reward_length"] = range_slice
    eval_multi_info["align_finish_max_range_reward"] = align_finish_max_range_reward
    eval_multi_info["align_finish_mean_range_reward"] = align_finish_mean_range_reward
    eval_multi_info["align_min_reward"] = align_min_reward
    eval_multi_info["align_mean_reward"] = align_mean_reward
    eval_multi_info["align_max_reward"] = align_max_reward

    if is_draw:
        plt.figure()
        plt.grid(True)
        plt.title("actor multi exp mean reward curves")
        plt.xlabel("time(s)")
        plt.ylabel("mean reward")
        plt.plot(range_time, np.array(align_mean_reward), alpha=0.75)
        plt.legend([f"{exp_name}"], loc="lower right")
        plt.fill_between(
            range_time,
            np.array(align_min_reward),
            np.array(align_max_reward),
            alpha=0.256,
        )
        plt.savefig(
            os.path.join(
                exp_dir,
                f"{exp_name}_{'_'.join(str(i) for i in exp_names)}_time_reward.png",
            ),
            dpi=256,
        )
        plt.close()

    with open(
        os.path.join(exp_dir, f"{exp_name}_eval_multi_actor_with_time_info.txt"),
        "w",
        encoding="utf-8",
    ) as fw:
        fw.write("multi exp actor reward summary:\n")
        for key, value in eval_multi_info.items():
            fw.write(f"{key}: {value}\n")
    return (
        range_time,
        (align_min_reward, align_mean_reward, align_max_reward),
        eval_multi_info,
    )


def eval_episode_multi(
    exp_dir, exp_name, finish_reward=None, is_draw=True, exp_indexs=5, range_slice=256
):
    # 一组实验，幕数：一个实验重复多次，在于后缀
    # exp_indexs是int，默认为range(exp_indexs)
    # exp_indexs是list、tuple，指定实验名称后缀，0和1都表示无后缀
    if isinstance(exp_indexs, int):
        exp_names = [exp_name]
        for i in range(2, exp_indexs + 1):
            exp_names.append(f"{exp_name}{i}")
    elif isinstance(exp_indexs, (tuple, list)):
        exp_names = []
        for index in exp_indexs:
            if isinstance(index, list):
                for item in index:
                    if item == 0 or item == 1:
                        exp_names.append(exp_name)
                    else:
                        exp_names.append(f"{exp_name}{item}")
            else:
                if index == 0 or index == 1:
                    exp_names.append(exp_name)
                else:
                    exp_names.append(f"{exp_name}{index}")
        exp_names = OrderedDict.fromkeys(exp_names).keys()
    else:
        raise TypeError(
            f"exp_indexs type need int、list、tuple， but got {type(exp_indexs)}"
        )

    eval_multi_info = {}
    multi_ep_times, multi_ep_rewards, multi_info = [], [], []
    for name in exp_names:
        ep_times, ep_rewards, ep_wins, info = eval_once(
            os.path.join(exp_dir, name), finish_reward=finish_reward, is_draw=is_draw
        )
        multi_ep_times.append(ep_times)
        multi_ep_rewards.append(ep_rewards)
        multi_info.append(info)
    align_episode = np.min([len(item) for item in multi_ep_rewards])
    multi_ep_times = np.stack(
        [np.array(item[:align_episode]) for item in multi_ep_times]
    )
    multi_ep_rewards = np.stack(
        [np.array(item[:align_episode]) for item in multi_ep_rewards]
    )
    align_episode = range(align_episode)

    align_finish_min_time, finish_min_flag, align_finish_min_range_reward = (
        np.inf,
        True,
        [],
    )
    align_finish_max_time, finish_max_flag, align_finish_max_range_reward = (
        np.inf,
        True,
        [],
    )
    align_finish_mean_time, finish_mean_flag, align_finish_mean_range_reward = (
        np.inf,
        True,
        [],
    )
    align_min_reward, align_max_reward, align_mean_reward = [], [], []
    for i in align_episode:
        range_data = multi_ep_rewards[:, i]
        _min_reward = np.min(range_data)
        _max_reward = np.max(range_data)
        _mean_reward = np.mean(range_data)
        align_min_reward.append(_min_reward)
        align_max_reward.append(_max_reward)
        align_mean_reward.append(_mean_reward)

        if finish_reward and finish_min_flag and _min_reward >= finish_reward:
            finish_min_flag = False
            align_finish_min_time = {
                "min_time": np.min(multi_ep_times[:, i]),
                "max_time": np.max(multi_ep_times[:, i]),
                "mean_time": np.mean(multi_ep_times[:, i]),
            }
            align_finish_min_range_reward = range_data

        if finish_reward and finish_max_flag and _max_reward >= finish_reward:
            finish_max_flag = False
            align_finish_max_time = {
                "min_time": np.min(multi_ep_times[:, i]),
                "max_time": np.max(multi_ep_times[:, i]),
                "mean_time": np.mean(multi_ep_times[:, i]),
            }
            align_finish_max_range_reward = range_data

        if finish_reward and finish_mean_flag and _mean_reward >= finish_reward:
            finish_mean_flag = False
            align_finish_mean_time = {
                "min_time": np.min(multi_ep_times[:, i]),
                "max_time": np.max(multi_ep_times[:, i]),
                "mean_time": np.mean(multi_ep_times[:, i]),
            }
            align_finish_mean_range_reward = range_data

    eval_multi_info["all_min_reward"] = np.min(
        [item["min_reward"] for item in multi_info]
    )
    eval_multi_info["all_max_reward"] = np.max(
        [item["max_reward"] for item in multi_info]
    )
    eval_multi_info["min_run_time"] = np.min([item["end_time"] for item in multi_info])
    eval_multi_info["max_run_time"] = np.max([item["end_time"] for item in multi_info])
    eval_multi_info["fast_finish_time"] = align_finish_max_time["min_time"]
    eval_multi_info["align_episode"] = align_episode
    eval_multi_info["align_finish_min_time"] = align_finish_min_time
    eval_multi_info["align_finish_max_time"] = align_finish_max_time
    eval_multi_info["align_finish_mean_time"] = align_finish_mean_time
    eval_multi_info["align_finish_min_range_reward"] = align_finish_min_range_reward
    eval_multi_info["align_finish_max_range_reward"] = align_finish_max_range_reward
    eval_multi_info["align_finish_mean_range_reward"] = align_finish_mean_range_reward
    eval_multi_info["align_min_reward"] = align_min_reward
    eval_multi_info["align_mean_reward"] = align_mean_reward
    eval_multi_info["align_max_reward"] = align_max_reward

    if is_draw:
        plt.figure()
        plt.grid(True)
        plt.title("actor multi exp mean reward curves")
        plt.xlabel("episode")
        plt.ylabel("mean reward")
        plt.plot(align_episode, np.array(align_mean_reward), alpha=0.75)
        plt.legend([f"{exp_name}"], loc="lower right")
        plt.fill_between(
            align_episode,
            np.array(align_min_reward),
            np.array(align_max_reward),
            alpha=0.256,
        )
        plt.savefig(
            os.path.join(
                exp_dir,
                f"{exp_name}_{'_'.join(str(i) for i in exp_names)}_episode_reward.png",
            ),
            dpi=256,
        )
        plt.close()

    with open(
        os.path.join(exp_dir, f"{exp_name}_eval_multi_actor_with_episode_info.txt"),
        "w",
        encoding="utf-8",
    ) as fw:
        fw.write("multi exp actor reward summary:\n")
        for key, value in eval_multi_info.items():
            fw.write(f"{key}: {value}\n")
    return (
        align_episode,
        (align_min_reward, align_mean_reward, align_max_reward),
        eval_multi_info,
    )


def eval_multi_with_multiactor(
    exp_dir,
    exp_name,
    finish_reward=None,
    exp_prefix: Union[List, Tuple] = (1, 2, 4, 8),
    exp_suffix: Union[List[Union[int, List]], Tuple[Union[int, List]]] = (5, 5, 5, 5),
    is_draw=True,
    range_slice=256,
    eval_func=eval_time_multi,
):
    # 多组实验：多组实验重复多次，在于前缀（汇总多组实验数据，对实验名字进行check）
    # exp_prefix是list、tuple，指定实验名称前缀，(1,2,4,8)代表1a、2a、4a、8a共计4组实验
    # exp_suffix指定实验后缀，代表这一组实验重复次数或重复实验的全部序号，第一维度与exp_prefix相同，第二维度可以为int、list、tuple
    if isinstance(exp_prefix, (tuple, list)):
        exp_actor_names = OrderedDict.fromkeys(exp_prefix).keys()
        assert len(exp_actor_names) == len(
            exp_suffix
        ), f"Inconsistent parameter length."
    else:
        raise TypeError(
            f"exp_prefix type need list、tuple， but got {type(exp_prefix)}"
        )

    multi_actor_exps = []
    for i, exp in enumerate(exp_actor_names):
        t, datas, infos = eval_func(
            exp_dir,
            f"{str(exp)}{exp_name}",
            finish_reward=finish_reward,
            exp_indexs=exp_suffix[i],
            is_draw=is_draw,
            range_slice=range_slice,
        )
        multi_actor_exps.append((t, datas, infos))

    if is_draw:
        plt.figure()
        plt.grid(True)
        plt.title("multi actor multi exp mean reward curves")
        plt.xlabel("time(s)")
        plt.ylabel("mean reward")
        for x, data, _ in multi_actor_exps:
            plt.plot(x, np.array(data[1]), alpha=0.75)
        plt.legend([f"{str(n)}actor" for n in exp_actor_names], loc="lower right")
        for x, data, _ in multi_actor_exps:
            plt.fill_between(x, np.array(data[0]), np.array(data[-1]), alpha=0.256)
        plt.savefig(
            os.path.join(
                exp_dir,
                f"{exp_name}_{'_'.join(str(i) for i in exp_actor_names)}_actor_reward.png",
            ),
            dpi=1024,
        )
        plt.close()


if __name__ == "__main__":
    args = parse_args()
    paths = args.exp
    exp_name = args.exp_name
    mode = args.mode
    finish_reward = args.finish_reward

    if mode == "once":
        eval_once(os.path.join(paths, exp_name), finish_reward=finish_reward)
    elif mode == "multi_time":
        exp_indexs = 5
        if args.exp_suffix:
            exec(f"_exp_indexs={args.exp_suffix}")
            exp_indexs = locals()["_exp_indexs"]
        eval_time_multi(
            paths, exp_name, finish_reward=finish_reward, exp_indexs=exp_indexs
        )
    elif mode == "multi_episode":
        exp_indexs = 5
        if args.exp_suffix:
            exec(f"_exp_indexs={args.exp_suffix}")
            exp_indexs = locals()["_exp_indexs"]
        eval_episode_multi(
            paths, exp_name, finish_reward=finish_reward, exp_indexs=exp_indexs
        )
    elif mode == "multi_actor":
        eval_func = eval_time_multi if args.eval_func == 0 else eval_episode_multi
        exp_prefix = [1, 2, 4, 8]
        if args.exp_prefix:
            exec(f"_exp_prefix={args.exp_prefix}")
            exp_prefix = locals()["_exp_prefix"]
        exp_suffix = [5, 5, 5, 5]
        if args.exp_suffix:
            exec(f"_exp_suffix={args.exp_suffix}")
            exp_suffix = locals()["_exp_suffix"]
        eval_multi_with_multiactor(
            paths,
            exp_name,
            finish_reward=finish_reward,
            exp_prefix=exp_prefix,
            exp_suffix=exp_suffix,
            eval_func=eval_func,
        )
    else:
        raise AttributeError(
            "eval mode only support [once, multi_time, multi_episode, multi_actor], please choice eval mode."
        )
