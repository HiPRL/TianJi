import sys
sys.path.insert(0, './../../../')



import unittest
import numpy as np
from env.gym_env.gym_env import GymEnv


class TestGymEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = GymEnv(gym_name="CartPole-v1")
    
    def tearDown(self) -> None:
        self.env.close()
    
    def test_gym_unit(self):
        self.assertEqual(self.env.action_dim, 2)
        self.assertEqual(self.env.status_dim, (4,))
        obs = self.env.reset()
        self.env.seed(np.random.randint(10))
        self.assertIsNotNone(obs)
        s_, r, done, info = self.env.step(0)
        self.assertEqual(s_.shape, (4,))
        self.assertIsInstance(r, float)
        self.assertIsInstance(done, bool)
        print(self.env)



if __name__ == '__main__':
    t = unittest.TestSuite()
    t.addTest(TestGymEnv('test_gym_unit'))
    run=unittest.TextTestRunner()
    run.run(t)