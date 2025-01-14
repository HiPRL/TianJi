# -*- coding: utf-8 -*-

# 一、environment（环境参数）
environment = dict(
    type = 'GymEnv',
    gym_name = "LunarLander-v2", # gym env selection
    render = False
)


# 二、experiment Parameters（实验配置）
exp = dict(
    max_step = 1000, # one of episode maximum step
    save_freq = 100, # frequency at which agents are save
)


# 三、Hyper Parameters（训练超参）
hyp = dict(
    LR = 0.0005,                                      # learning rate
    GAMMA = 0.99,                                    # reward discount
    EPSILON = 0.02,                                  # greedy policy
    TARGET_REPLACE_ITER = 128,                      # target update frequency
    warmup_size = 2000,
    warmup_full_random = True,
    buffer_size = 5000000,
    batch_size = 64
)


# 四、agent（搭建智能体）
model = dict(
    type='MLP',
    c1=8, # state dim
    c2=4, # action dim
    dueling=True
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
        send_interval = 16,
        exit_val = dict(
            learn_step = 200000,
        ),
    ),
    actor_cfg = dict(
        num = 1,
        send_size = 32,
    ),
    buffer_cfg = dict(
        global_buffer = dict(
            type='StepBuffer',
            max_size=10000
        ),
        send_size = 64
    )
)