# -*- coding: utf-8 -*-

# 一、environment（环境参数）
environment = dict(
    type = 'GymEnv',
    gym_name = "AssaultNoFrameskip-v4", # gym env selection
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
    save_freq = 100000, # frequency at which agents are save
    eval_step = 100000 # evaluate episode interval
)

ACTOR_NUM = 4

# 三、Hyper Parameters（训练超参）
hyp = dict(
    lr = 0.00025,
    clip_param = 0.1,
    gamma = 0.99,
    gae_lambda = 0.95,
    update_step = 4,
    batch_size = ACTOR_NUM,
    step_len = 200,
    warmup_size = ACTOR_NUM,
    buffer_size = ACTOR_NUM,
)


# 四、agent（搭建智能体）
model = dict(
    type='PPOAtari',
    state_dim=4,
    action_dim=7,
)
embryo = dict(
    type='PPOHead',
    model=model,
    hyp=hyp
)
agent = dict(
    type='PPO',
    embryo=embryo,
    alg_type='on-policy'
)

# 五、hook（算法hook、辅助功能hook...）
hooks = [
    dict(type='PPOHook'),
]


# 六、分布式参数
parallel_parameters = dict(
    global_cfg = dict(
        use_group_parallel = True,
        group_num = 2,
        save_freq = 10,
        exit_val = dict(
            fusion_step = 10000, # 84耗时40000
        ),
    ),
    learner_cfg = dict(
        num = 1,
        cores = 16,
        send_interval = 10000000,
        send_root_interval = 1,
        finish_reward = 12,
        exit_val = dict(
            learn_step = 10000000,
        ),
    ),
    actor_cfg = dict(
        num = ACTOR_NUM,
        send_size = 1,
    ),
    buffer_cfg = dict(
        global_buffer = dict(
            type='MultiStepBuffer',
            max_size=1000,
            step_limit=2048
        ),
        send_size = 1
    )
)
