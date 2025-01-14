# -*- coding: utf-8 -*-

# 一、environment（环境参数）
environment = dict(
    type = 'GymEnv',
    gym_name = "PongNoFrameskip-v4", # gym env selection
    render = False,
    wrappers = [dict(type="MonitorEnv"),
                dict(type="AtariOriginalReward"),
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
    save_freq = 5000, # frequency at which agents are save
    eval_step = 10000000 # evaluate episode interval
)


# 三、Hyper Parameters（训练超参）
hyp = dict(
    LR = 0.00025,                                      # learning rate
    GAMMA = 0.99,                                    # reward discount
    EPSILON = 0.02,                                  # greedy policy
    TARGET_REPLACE_ITER = 100,                      # target update frequency
    warmup_size = 625,
    buffer_size = 25000,
    batch_size = 16
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
    global_cfg = dict(
        use_group_parallel = True,
        group_num = 1,
        save_freq = 5000, # frequency at which agents are save
        exit_val = dict(
            fusion_step = 80000, 
        ),
    ),
    learner_cfg = dict(
        num = 1,
        cores = 16,
        send_interval = 1000000,    
        send_root_interval = 12,
        finish_reward = 12,
        exit_val = dict(
            learn_step = 10000000,
        ),
    ),
    actor_cfg = dict(
        num = 4,
        send_size = 4,
    ),
    buffer_cfg = dict(
        global_buffer = dict(
            type='StepBuffer',
            max_size=1000
        ),
        cores = 1,
        send_size = 4
    )
)
