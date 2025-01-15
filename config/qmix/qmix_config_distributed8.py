# -*- coding: utf-8 -*-

# 一、environment（环境参数）
environment = dict(
    type="StarCraft2Env",
    map_name="8m",  # StarCraft2 env selection
    difficulty="7",
    render=False,
    seed=None,
)


# 二、experiment Parameters（实验配置）
exp = dict(
    train_steps=1000000,  # number of steps
    max_step=120,  # one of episode maximum step
    save_freq=100,  # frequency at which agents are save
    eval_times=20,  # number of eval times
    eval_step=100,  # evaluate episode interval
)


# 三、Hyper Parameters（训练超参）
hyp = dict(
    LR=0.0001,  # learning rate
    GAMMA=0.99,  # reward discount
    EPSILON=0.99,  # greedy policy
    max_decay_step=2500,
    TARGET_REPLACE_ITER=64,  # target update frequency
    warmup_size=32,
    buffer_size=5000,
    batch_size=32,
    double_q=True,
    clip_grad_norm=10,
)


# 四、agent（搭建智能体）
agent_num = 8
model = dict(
    type="GRU",
    c1=102,  # obs dim + agent dim + action dim
    c2=14,  # action dim
)
mixer_model = dict(
    type="MixerNet",
    agent_num=agent_num,
    state_shape=168,  # state dim
)
embryo = dict(type="MixerHead", model=model, mixer_model=mixer_model, hyp=hyp)
agent = dict(type="QMIX", embryo=embryo)


# 五、hook（算法hook、辅助功能hook...）
hooks = [
    dict(type="QmixHook"),
    # dict(type='TensorboardHook')
]


# 六、分布式参数
parallel_parameters = dict(
    learner_cfg=dict(
        num=1,
        cores=1,
        send_interval=8,
        exit_val=dict(
            learn_step=32000,
        ),
    ),
    actor_cfg=dict(
        num=8,
        send_size=4,
    ),
    buffer_cfg=dict(
        global_buffer=dict(type="EpisodeBuffer", max_size=10000, episode_limit=120),
        send_size=32,
    ),
)
