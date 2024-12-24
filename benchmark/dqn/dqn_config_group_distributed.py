# -*- coding: utf-8 -*-

# 一、environment（环境参数）
environment = dict(
    type = 'GymEnv',
    gym_name = "CartPole-v1", # gym env selection
    render = False
)


# 二、experiment Parameters（实验配置）
exp = dict(
    max_step = 500, # one of episode maximum step
    save_freq = 100, # frequency at which agents are save
    eval_step = 100 # evaluate episode interval
)


# 三、Hyper Parameters（训练超参）
hyp = dict(
    LR = 0.0005,                                      # learning rate
    GAMMA = 0.99,                                    # reward discount
    EPSILON = 0.02,                                  # greedy policy
    TARGET_REPLACE_ITER = 200,                      # target update frequency
    warmup_size = 1000,
    warmup_full_random = True,
    buffer_size = 50000,
    batch_size = 256
)


# 四、agent（搭建智能体）
model = dict(
    type='MLP',
    c1=4, # state dim
    c2=2, # action dim
    dueling=True,
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
    dict(type='DQNCartPoleV1Hook'),
]


# 六、分布式参数
parallel_parameters = dict(
    global_cfg = dict(
        use_group_parallel = True,
        group_num = 1
    ),
    learner_cfg = dict(
        num = 1,
        cores = 1,
        send_interval = 128,
        send_root_interval = 16,
        finish_reward = 300,
        exit_val = dict(
            learn_step = 10000,
        ),
        eval_val = dict(
            eval_count = 100,
            reward_opt = dict(
                step_opt = "mean",
                episode_opt = "sum"
            )
        )
    ),
    actor_cfg = dict(
        num = 4,
        send_size = 32,
    ),
    buffer_cfg = dict(
        global_buffer = dict(
            type='StepBuffer',
            max_size=10000
        ),
        cores = 1,
        send_size = 32
    )
)