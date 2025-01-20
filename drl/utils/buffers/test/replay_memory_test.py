import sys
sys.path.insert(0, './../../../../')

from drl.utils.buffers import Experience, ReplayMemory
import unittest
import numpy as np




class ReplayMemoryTest(unittest.TestCase):
    def test_RM_unit(self) -> None:
        rm = ReplayMemory(512)
        self.assertEqual(len(rm), 0)
        for i in range(1000):
            rm.push(i, 0, 0, 0, 0, 0, 0)
        self.assertTrue(rm.is_overflow)
        self.assertEqual(len(rm), 512)
        p_t = rm.pop(32)
        self.assertEqual(len(rm), 512-32)
        self.assertEqual(len(p_t), 32)
        self.assertFalse(rm.is_overflow)
        self.assertEqual(rm.size, 512)
        self.assertEqual(rm.index, 1000 % 512)
        rm.clear()
        self.assertEqual(len(rm), 0)
        
        
        
        

if __name__ == '__main__':
    t = unittest.TestSuite()
    t.addTest(ReplayMemoryTest('test_RM_unit'))
    run=unittest.TextTestRunner()
    run.run(t)