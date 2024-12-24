# -*- coding: utf-8 -*-

# 一、environment（环境参数）
environment = dict(
    type = 'MultiagentParticleEnv',
    scenario_name = "simple_tag", # MPE scenarios selection
    render = True,
)


# 二、experiment Parameters（实验配置）
exp = dict(
    episodes = 25000, # number of episodes
    max_step = 25, # one of episode maximum step
    save_freq = 100, # frequency at which agents are save
    eval_step = 5 # evaluate episode interval
)


# 三、Hyper Parameters（训练超参）
hyp = dict(
    actor_lr = 0.01,
    critic_lr = 0.01,
    gamma = 0.95,
    tau = 0.01,
    batch_size = 64,
    warmup_size = 2048,
    buffer_size = 1000000,
    epsilon = 0.97,
    target_replace_iter = 100
)


# 四、agent（搭建智能体）
model = dict(
    type='ActorCritic',
    actor_state_dim=20, # state dim
    actor_action_dim=5, # action dim
    critic_dim=100,
    mlp_dim=64
)
embryo = dict(
    type='MADDPGHead',
    model=model,
    hyp=hyp
)
agent = dict(
    type='MADDPG',
    embryo=embryo,
    agent_num=4
)


# 五、hook（算法hook、辅助功能hook...）
hooks = [
    dict(type='MADDPGHook'),
    dict(type='TensorboardHook')
]

