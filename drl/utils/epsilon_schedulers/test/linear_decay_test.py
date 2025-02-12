import sys
sys.path.insert(0, './../../../../')

import unittest
import numpy as np
from drl.utils.epsilon_schedulers.linear_decay import LinearDecayEpsilonScheduler

class LinearDecayEpsilonSchedulerTest(unittest.TestCase):

    def setUp(self):
        # 固定随机数种子以便测试结果可重复
        np.random.seed(0)

    def test_initialization(self):
        """ 测试初始化参数是否正确 """
        scheduler = LinearDecayEpsilonScheduler(start_epsilon=1.0, end_epsilon=0.1, max_step=100, min_step=10)
        self.assertEqual(scheduler.start_epsilon, 1.0)
        self.assertEqual(scheduler.end_epsilon, 0.1)
        self.assertEqual(scheduler.max_step, 100)
        self.assertEqual(scheduler.min_step, 10)

    def test_initialization_with_invalid_parameters(self):
        """ 测试初始化时非法参数是否引发异常 """
        with self.assertRaises(AssertionError):
            LinearDecayEpsilonScheduler(start_epsilon=-0.1, end_epsilon=0.1, max_step=100, min_step=10)  # start_epsilon < 0
        with self.assertRaises(AssertionError):
            LinearDecayEpsilonScheduler(start_epsilon=1.0, end_epsilon=1.1, max_step=100, min_step=10)  # end_epsilon > 1
        with self.assertRaises(AssertionError):
            LinearDecayEpsilonScheduler(start_epsilon=1.0, end_epsilon=0.1, max_step=10, min_step=20)  # min_step >= max_step
        with self.assertRaises(AssertionError):
            LinearDecayEpsilonScheduler(start_epsilon=1.0, end_epsilon=0.1, max_step=10, min_step=0)  # min_step <= 0

    def test_step_method(self):
        """ 测试 step 方法的行为 """
        scheduler = LinearDecayEpsilonScheduler(start_epsilon=1.0, end_epsilon=0.1, max_step=100, min_step=10)

        # 情况 1: x < min_step
        self.assertAlmostEqual(scheduler.step(5), 1.0)  # 应该返回 start_epsilon

        # 情况 2: x > max_step
        self.assertAlmostEqual(scheduler.step(150), 0.1)  # 应该返回 end_epsilon

        # 情况 3: min_step <= x <= max_step
        self.assertAlmostEqual(scheduler.step(10), 1.0)  # 刚好等于 min_step
        self.assertAlmostEqual(scheduler.step(100), 0.1)  # 刚好等于 max_step
        self.assertAlmostEqual(scheduler.step(55), 0.55)  # 中间值，线性插值

    def test_step_method_edge_cases(self):
        """ 测试 step 方法在极端情况下的行为 """
        scheduler = LinearDecayEpsilonScheduler(start_epsilon=1.0, end_epsilon=0.1, max_step=10, min_step=10)

        # 当 min_step == max_step 时，应该返回 end_epsilon
        self.assertAlmostEqual(scheduler.step(10), 0.1)
        self.assertAlmostEqual(scheduler.step(5), 1.0)  # 小于 min_step
        self.assertAlmostEqual(scheduler.step(15), 0.1)  # 大于 max_step

    def test_explore_method(self):
        """ 测试 explore 方法的行为 """
        scheduler = LinearDecayEpsilonScheduler(start_epsilon=1.0, end_epsilon=0.1, max_step=100, min_step=10)
        random_action, greedy_action = self._mock_actions()

        # 情况 1: x < min_step，总是返回随机动作
        action = scheduler.explore(5, random_action, greedy_action)
        self.assertEqual(action, 'random')

        # 情况 2: x > max_step，总是返回贪婪动作
        action = scheduler.explore(150, random_action, greedy_action)
        self.assertEqual(action, 'greedy')

        # 情况 3: min_step <= x <= max_step，按照 epsilon 概率选择动作
        trials = 10000
        explore_count = 0
        exploit_count = 0

        for _ in range(trials):
            action = scheduler.explore(55, random_action, greedy_action)
            if action == 'random':
                explore_count += 1
            else:
                exploit_count += 1

        explore_rate = explore_count / trials
        exploit_rate = exploit_count / trials

        # 验证探索率大约等于 epsilon 值 (0.55)，允许一定误差范围
        self.assertAlmostEqual(explore_rate, 0.55, delta=0.01)
        self.assertAlmostEqual(exploit_rate, 0.45, delta=0.01)

    def test_value_property(self):
        """ 测试 value 属性的行为 """
        scheduler = LinearDecayEpsilonScheduler(start_epsilon=1.0, end_epsilon=0.1, max_step=100, min_step=10)
        random_action, greedy_action = self._mock_actions()

        # 使用 explore 方法更新 epsilon_value
        scheduler.explore(5, random_action, greedy_action)
        self.assertAlmostEqual(scheduler.value, 1.0)  # 对应 step(5)

        scheduler.explore(150, random_action, greedy_action)
        self.assertAlmostEqual(scheduler.value, 0.1)  # 对应 step(150)

        scheduler.explore(55, random_action, greedy_action)
        self.assertAlmostEqual(scheduler.value, 0.55)  # 对应 step(55)

    def _mock_actions(self):
        """ 提供模拟的随机和贪婪动作函数 """
        return lambda: 'random', lambda: 'greedy'


if __name__ == '__main__':
    unittest.main()