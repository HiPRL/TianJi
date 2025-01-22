# -*- coding: utf-8 -*-

# 一、environment（环境参数）
environment = dict(
    type = 'GymEnv',
    gym_name = "CartPole-v1", # gym env selection
    render = False
)


# 二、experiment Parameters（实验配置）
exp = dict(
    train_steps = 2500000, # number of steps
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
    update_step = 80,
    batch_size = 1,
    step_len = 2048,
    warmup_size = 1,
    buffer_size = 1000,
)


# 四、agent（搭建智能体）
model = dict(
    type="PPOActorCritic",
    state_dim=4,
    action_dim=2,
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