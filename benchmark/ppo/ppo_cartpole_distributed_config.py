# -*- coding: utf-8 -*-

# 一、environment（环境参数）
environment = dict(
    type = 'GymEnv',
    gym_name = "CartPole-v1", # gym env selection
    render = False
)


# 二、experiment Parameters（实验配置）
exp = dict(
    train_steps = 25000, # number of steps
    max_step = 500, # one of episode maximum step
    save_freq = 100, # frequency at which agents are save
    eval_step = 100 # evaluate episode interval
)


# 三、Hyper Parameters（训练超参）
hyp = dict(
    lr = 0.0001,
    clip_param = 0.2,
    gamma = 0.99,
    gae_lambda = 0.95,
    update_step = 10,
    batch_size = 1,
    step_len = 2048,
    warmup_size = 1,
    buffer_size = 64,
)


# 四、agent（搭建智能体）
model = dict(
    type='PPOActorCritic',
    actor_state_dim=4,
    actor_action_dim=2,
    critic_state_dim=4,
    mlp_dim=64,
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
    learner_cfg = dict(
        num = 1,
        cores = 1,
        send_interval = 1,
        finish_reward = 300,
        exit_val = dict(
            learn_step = 700,
        ),
    ),
    actor_cfg = dict(
        num = 1,
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