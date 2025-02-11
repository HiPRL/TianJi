import sys
sys.path.insert(0, './../../../')

import unittest
import numpy as np
from env.starcraft2_env.starcraft2_env import StarCraft2Env


class TestStarCraft2Env(unittest.TestCase):
    def setUp(self) -> None:
        self.env = StarCraft2Env(map_name = "8m",difficulty = '7',render = False)
    
    def tearDown(self) -> None:
        self.env.close()
    
    def test_sc2_unit(self):
        self.assertEqual(self.env.agent_num, 8)
        self.assertEqual(self.env.enemy_num, 8)
        self.assertEqual(self.env.status_dim, (168))
        self.assertEqual(self.env.observations_dim, 80)
        self.assertEqual(self.env.action_dim, 14)
        self.assertEqual(self.env.episode_limit, 120)
        self.env.reset()
        self.env.seed(np.random.randint(10))

        actions = []
        for agent_id in range(self.env.agent_num):
            avail_actions = self.env.env.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = np.random.choice(avail_actions_ind)
            actions.append(action)

        s_, r, done, info = self.env.step(actions)
        self.assertEqual(s_.shape, (168,))
        self.assertIsInstance(r, float)
        self.assertIsInstance(done, bool)



if __name__ == '__main__':
    t = unittest.TestSuite()
    t.addTest(TestStarCraft2Env('test_sc2_unit'))
    run=unittest.TextTestRunner()
    run.run(t)