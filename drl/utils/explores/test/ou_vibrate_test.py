import sys
sys.path.insert(0, './../../../../')

import unittest
import torch
import numpy as np
from scipy.stats import kstest, norm

from drl.utils.explores.ou_vibrate import OrnsteinUhlenbeckVibrateExplore

class OrnsteinUhlenbeckVibrateExploreTest(unittest.TestCase):

    def setUp(self):
        # 固定随机数种子以便测试结果可重复
        np.random.seed(0)

    def test_initialization(self):
        """ 测试初始化参数 """
        # 默认参数
        ouve = OrnsteinUhlenbeckVibrateExplore()
        self.assertEqual(ouve.mean, 0.0)
        self.assertEqual(ouve.theta, 0.15)
        self.assertEqual(ouve.sigma, 0.3)
        self.assertIsNone(ouve.ou_value)

        # 自定义参数
        ouve_custom = OrnsteinUhlenbeckVibrateExplore(mean=1.0, theta=0.2, sigma=0.5)
        self.assertEqual(ouve_custom.mean, 1.0)
        self.assertEqual(ouve_custom.theta, 0.2)
        self.assertEqual(ouve_custom.sigma, 0.5)
        self.assertIsNone(ouve_custom.ou_value)

    def test_sample_with_numpy_array(self):
        """ 测试 NumPy 数组输入 """
        ouve = OrnsteinUhlenbeckVibrateExplore(mean=0.0, theta=0.15, sigma=0.3)

        input_data = np.array([1.0, 2.0, 3.0])
        result = ouve.sample(input_data)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (3,))
        self.assertIsNotNone(ouve.ou_value)

    def test_sample_with_pytorch_tensor(self):
        """ 测试 PyTorch 张量输入 """
        ouve = OrnsteinUhlenbeckVibrateExplore(mean=0.0, theta=0.15, sigma=0.3)

        input_data = torch.tensor([1.0, 2.0, 3.0])
        result = ouve.sample(input_data)
        self.assertTrue(isinstance(result, np.ndarray))  # 输出始终是 NumPy 数组
        self.assertEqual(result.shape, (3,))
        self.assertIsNotNone(ouve.ou_value)

    def test_invalid_input_types(self):
        """ 测试无效输入类型 """
        ouve = OrnsteinUhlenbeckVibrateExplore(mean=0.0, theta=0.15, sigma=0.3)

        # 标量输入
        with self.assertRaises(TypeError):
            ouve.sample(1.0)

        # 列表输入
        with self.assertRaises(TypeError):
            ouve.sample([1.0, 2.0, 3.0])

        # 字符串输入
        with self.assertRaises(TypeError):
            ouve.sample("invalid")

    def test_ou_process_logic(self):
        """ 测试 OU 过程逻辑 """
        ouve = OrnsteinUhlenbeckVibrateExplore(mean=0.0, theta=0.15, sigma=0.3)

        # 初始化噪声值
        ouve.sample(np.array([1.0, 2.0, 3.0]))
        initial_noise = ouve.ou_value.copy()

        # 收集多次采样的随机成分
        random_components = []
        for _ in range(1000):  # 收集 100 次采样
            ouve.sample(np.array([1.0, 2.0, 3.0]))
            updated_noise = ouve.ou_value
            delta = updated_noise - initial_noise
            expected_delta = ouve.theta * (ouve.mean - initial_noise)
            random_component = delta - expected_delta
            random_components.extend(random_component)  # 扩展为一维数组
            initial_noise = updated_noise.copy()  # 更新初始噪声

        # 验证随机成分的均值和标准差
        mean_random = np.mean(random_components)
        std_random = np.std(random_components)
        self.assertAlmostEqual(mean_random, 0, delta=0.05)  # 均值接近 0
        self.assertAlmostEqual(std_random, ouve.sigma, delta=0.05)  # 标准差接近 sigma

        # 使用 KS 检验验证随机成分是否符合正态分布
        ks_stat, p_value = kstest(random_components, 'norm', args=(0, ouve.sigma))
        self.assertGreater(p_value, 0.05)  # 如果 p 值 > 0.05，则认为数据符合正态分布

    def test_multiple_calls(self):
        """ 测试多次调用 sample 方法 """
        ouve = OrnsteinUhlenbeckVibrateExplore(mean=0.0, theta=0.15, sigma=0.3)

        # 多次调用 sample 方法
        results = []
        for _ in range(5):
            results.append(ouve.sample(np.array([1.0, 2.0, 3.0])))

        # 验证每次的结果不同
        for i in range(len(results) - 1):
            self.assertFalse(np.allclose(results[i], results[i + 1]))

if __name__ == '__main__':
    unittest.main()
