# -*- coding: utf-8 -*-

# 一、environment（环境参数）
environment = dict(
    type = 'PredatorEnv',
    scenario_name = 'endless3'
)


# 二、experiment Parameters（实验配置）
exp = dict(
    train_steps = 100000, # number of steps
    max_step = 200, # one of episode maximum step
    save_freq = 100, # frequency at which agents are save
    eval_times = 20, # number of eval times
    eval_step = 100 # evaluate episode interval
)


# 三、Hyper Parameters（训练超参）
hyp = dict(
    LR = 0.0001,                                     # learning rate
    GAMMA = 0.99,                                    # reward discount
    EPSILON = 0.99,                                   # greedy policy
    max_decay_step = 10000,
    TARGET_REPLACE_ITER = 30,                     # target update frequency
    warmup_size = 32,
    buffer_size = 10000,
    batch_size = 32,
    double_q = True,
    clip_grad_norm = 10
)


# 四、agent（搭建智能体）
agent_num = 2
model = dict(
    type='MLP',
    c1=9, # obs dim + agent dim + action dim
    c2=5, # action dim
)
mixer_model = dict(
    type='MixerNet',
    agent_num=agent_num,
    state_shape=6, # state dim
)
embryo = dict(
    type='PredatorMixerHead',
    model=model,
    mixer_model=mixer_model,
    hyp=hyp
)
agent = dict(
    type='PredatorQMIX',
    embryo=embryo
)


# 五、hook（算法hook、辅助功能hook...）
hooks = [
    dict(type='QmixPredatorHook'),
    dict(type='TensorboardHook')
]