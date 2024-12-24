# -*- coding: utf-8 -*-

# 一、environment（环境参数）
environment = dict(
    type = 'GymEnv',
    gym_name = "PongNoFrameskip-v4", # gym env selection
    render = False,
    wrappers = [dict(type="MonitorEnv"),
                dict(type="NoopResetEnv"),
                dict(type="MaxAndSkipEnv"),
                dict(type="EpisodicLifeEnv"),
                dict(type="FireResetEnv"),
                dict(type="WarpFrame"),
                dict(type="ScaledFloatFrame"),
                dict(type="ClipRewardEnv"),
                dict(type="FrameStackOrder", k=4, obs_format='NCHW')]
)


# 二、experiment Parameters（实验配置）
exp = dict(
    train_steps = 10000000, # number of steps
    save_freq = 100000, # frequency at which agents are save
    eval_step = 100000 # evaluate episode interval
)


# 三、Hyper Parameters（训练超参）
hyp = dict(
    LR = 0.0003,                                      # learning rate
    GAMMA = 0.99,                                    # reward discount
    EPSILON = 0.02,                                  # greedy policy
    TARGET_REPLACE_ITER = 200,                      # target update frequency
    warmup_size = 10000,
    buffer_size = 1000000,
    batch_size = 32
)


# 四、agent（搭建智能体）
model = dict(
    type='AtariModel',
    act_dim=6, # action dim
)
embryo = dict(
    type='DQNHead',
    model=model,
    hyp=hyp
)
agent = dict(
    type='DQN',
    embryo=embryo
)


# 五、hook（算法hook、辅助功能hook...）
hooks = [
    dict(type='DQNHook'),
]


# 六、分布式参数
parallel_parameters = dict(
    learner_cfg = dict(
        num = 1,
        send_interval = 128,
        finish_reward = 21,
        exit_val = dict(
            learn_step = 10000000,
        ),
    ),
    actor_cfg = dict(
        num = 2,
        send_size = 4,
    ),
    buffer_cfg = dict(
        global_buffer = dict(
            type='StepBuffer',
            max_size=10000
        ),
        send_size = 8
    )
)