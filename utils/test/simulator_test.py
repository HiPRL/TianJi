# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, './../..') 
import unittest
import numpy as np

from utils import Simulator  


STOP_STEP = 3


class TestSimulator(unittest.TestCase):
    """
    测试一个episode,assert _episode, episode_step, reward, scalar_buffer;
    测试run, 正常关闭
    """
    def setUp(self):
        self.agent = DummyAgent(alg_type="off-policy")
        self.env = DummyEnv()
        self.cfg = DummyCfg()
        self.simulator = Simulator(self.agent, self.env, self.cfg)
        self.simulator.scalar_buffer = DummyScalarBuffer()

        self.simulator._episode = 0
        self.simulator._train_step = 0

        self.simulator.call_hook = lambda hook, *args, **kwargs: []
        self.simulator.reduce_factor = lambda factor: ([], [])

    def test_train_episode(self):
        self.simulator.train_episode()
        self.assertEqual(self.simulator._episode, 1)
        self.assertEqual(self.simulator.episode_step, STOP_STEP)
        self.assertAlmostEqual(self.simulator.episode_reward, 2.0)
        self.assertTrue(len(self.simulator.scalar_buffer.updates) > 0)

    def test_run(self):
        self.simulator.run()
        self.assertTrue(self.agent.resumed)
        self.assertTrue(self.env.closed)


class DummyAgent:
    def __init__(self, alg_type="off-policy"):
        self.alg_type = alg_type
        self.learn_step = 1
        self.alg_scalar_data = {"dummy": 0.123}
        self.resumed = False

    def policy(self, *args):
        return 0

    def learn(self, episode, status, actions, reward, status_, is_over, learn_factor):
        return 0

    def predict(self, *args):
        return 0

    def resume(self, resume_path):
        self.resumed = True


class DummyEnv:
    def __init__(self):
        self.step_count = 0
        self.closed = False

    def reset(self):
        self.step_count = 0
        return 0

    def render(self):
        pass

    def step(self, action):
        self.step_count += 1
        if self.step_count >= STOP_STEP:
            return 1, 1.0, True, {}
        else:
            return 0, 0.5, False, {}

    def close(self):
        self.closed = True


class DummyCfg:
    def __init__(self):
        self.resume = "dummy_resume"
        self.exp = dict(
            train_steps = 25000, # number of steps
            max_step = 500, # one of episode maximum step
            save_freq = 100, # frequency at which agents are save
            eval_step = 100 # evaluate episode interval
        )


class DummyScalarBuffer:
    def __init__(self):
        self.updates = []

    def update(self, data, index=None):
        self.updates.append((data, index))



if __name__ == '__main__':
    t = unittest.TestSuite()
    t.addTest(TestSimulator('test_train_episode'))
    t.addTest(TestSimulator('test_run'))
    r = unittest.TextTestRunner()
    r.run(t)
