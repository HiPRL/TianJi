import sys
sys.path.insert(0, './../../../../')

import unittest
import numpy as np
from drl.utils.epsilon_schedulers.constant import ConstantEpsilonScheduler

class ConstantEpsilonSchedulerTest(unittest.TestCase):

    def setUp(self):
        # 固定随机数种子以便测试结果可重复
        np.random.seed(0)

    def test_step_method(self):
        """ 测试step方法是否按预期工作（在此实现中不执行任何操作） """
        scheduler = ConstantEpsilonScheduler(0.1)
        try:
            scheduler.step(1)  # 尝试调用step方法
            self.assertTrue(True, "step方法可以被调用且不会抛出异常")
        except NotImplementedError as e:
            self.fail(f"step方法不应抛出NotImplementedError: {e}")
            
    def test_explore_with_epsilon_1(self):
        """ 测试当epsilon值为1时，总是进行探索 """
        scheduler = ConstantEpsilonScheduler(1.0)
        random_action, greedy_action = self._mock_actions()
        
        action, is_greedy = scheduler.explore(None, random_action, greedy_action)
        
        self.assertFalse(is_greedy)  # 应该总是返回False，因为总是在探索
        self.assertEqual(action, 'random')  # 假设随机动作返回'random'

    def test_explore_with_epsilon_0(self):
        """ 测试当epsilon值为0时，总是进行利用 """
        scheduler = ConstantEpsilonScheduler(0.0)
        random_action, greedy_action = self._mock_actions()
        
        action, is_greedy = scheduler.explore(None, random_action, greedy_action)
        
        self.assertTrue(is_greedy)  # 应该总是返回True，因为总是在利用
        self.assertEqual(action, 'greedy')  # 假设贪婪动作返回'greedy'

    def test_explore_with_epsilon_0_2(self):
        """ 测试当epsilon值为0.2时，按照给定概率进行探索或利用 """
        epsilon = 0.2
        scheduler = ConstantEpsilonScheduler(epsilon)
        random_action, greedy_action = self._mock_actions()

        explore_count = 0
        exploit_count = 0
        trials = 10000  # 进行大量试验以验证概率接近预期值
        
        for _ in range(trials):
            action, is_greedy = scheduler.explore(None, random_action, greedy_action)
            if not is_greedy:
                explore_count += 1
            else:
                exploit_count += 1

        explore_rate = explore_count / trials
        exploit_rate = exploit_count / trials

        # 验证探索率大约等于epsilon值，考虑到统计波动允许一定误差范围
        self.assertAlmostEqual(explore_rate, epsilon, delta=0.01)
        self.assertAlmostEqual(exploit_rate, 1 - epsilon, delta=0.01)

    def _mock_actions(self):
        """ 提供模拟的随机和贪婪动作函数 """
        return lambda: 'random', lambda: 'greedy'


if __name__ == '__main__':
    unittest.main()