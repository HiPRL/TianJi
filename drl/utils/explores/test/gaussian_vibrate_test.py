import sys
sys.path.insert(0, './../../../../')


import unittest
import torch
import numpy as np
from drl.utils.explores.gaussian_vibrate import GaussianVibrateExplore

class GaussianVibrateExploreTest(unittest.TestCase):

    def setUp(self):
        # 固定随机数种子以便测试结果可重复
        torch.manual_seed(0)
        np.random.seed(0)

    def test_initialization(self):
        """ 测试初始化参数 """
        gve = GaussianVibrateExplore()
        self.assertEqual(gve.mean, 0.0)
        self.assertEqual(gve.std, 1e-2)
        self.assertIsNone(gve.low)
        self.assertIsNone(gve.high)

    def test_sample_with_numpy_array(self):
        """ 测试 NumPy 数组输入 """
        gve = GaussianVibrateExplore(mean=0.0, std=1e-2, low=None, high=None)

        input_data = np.array([1.0, 2.0, 3.0])
        result = gve.sample(input_data)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3,))
        for i in range(3):
            self.assertAlmostEqual(float(result[i]), input_data[i], delta=0.03)

    def test_sample_with_pytorch_tensor(self):
        """ 测试 PyTorch 张量输入 """
        gve = GaussianVibrateExplore(mean=0.0, std=1e-2, low=None, high=None)

        input_data = torch.tensor([1.0, 2.0, 3.0])
        result = gve.sample(input_data)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3,))
        for i in range(3):
            self.assertAlmostEqual(float(result[i]), float(input_data[i]), delta=0.03)

    def test_invalid_input_types(self):
        """ 测试无效输入类型 """
        gve = GaussianVibrateExplore(mean=0.0, std=1e-2, low=None, high=None)

        # 标量输入
        with self.assertRaises(TypeError):
            gve.sample(1.0)

        # 列表输入
        with self.assertRaises(TypeError):
            gve.sample([1.0, 2.0, 3.0])

        # 字符串输入
        with self.assertRaises(TypeError):
            gve.sample("invalid")

    def test_callable_input(self):
        """ 测试可调用对象作为输入 """
        gve = GaussianVibrateExplore(mean=0.0, std=1e-2, low=None, high=None)

        def get_input():
            return torch.tensor([1.0, 2.0, 3.0])

        result = gve.sample(get_input)
        self.assertTrue(isinstance(result, torch.Tensor))
        self.assertEqual(result.shape, (3,))
        for i in range(3):
            self.assertAlmostEqual(float(result[i]), [1.0, 2.0, 3.0][i], delta=0.03)


if __name__ == '__main__':
    unittest.main()