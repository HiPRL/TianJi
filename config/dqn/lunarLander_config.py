# -*- coding: utf-8 -*-

# 一、environment（环境参数）
environment = dict(
    type="GymEnv", gym_name="LunarLander-v2", render=False  # gym env selection
)


# 二、experiment Parameters（实验配置）
exp = dict(
    episodes=25000,  # number of episodes
    max_step=1000,  # one of episode maximum step
    save_freq=100,  # frequency at which agents are save
)


# 三、Hyper Parameters（训练超参）
hyp = dict(
    LR=0.0005,  # learning rate
    GAMMA=0.99,  # reward discount
    EPSILON=0.02,  # greedy policy
    TARGET_REPLACE_ITER=128,  # target update frequency
    warmup_size=2000,
    warmup_full_random=True,
    buffer_size=50000,
    batch_size=64,
)


# 四、agent（搭建智能体）
model = dict(type="MLP", c1=8, c2=4, dueling=True)  # state dim  # action dim
embryo = dict(type="DQNHead", model=model, hyp=hyp)
agent = dict(type="DQN", embryo=embryo)


# 五、hook（算法hook、辅助功能hook...）
hooks = [dict(type="DQNHook"), dict(type="TensorboardHook")]
