import sys
sys.path.insert(0, './../../../../')

from drl.utils.buffers import MultiStepBuffer
import unittest
import numpy as np
import random




class MultiStepBufferTest(unittest.TestCase):
    def test_msb_unit(self) -> None:
        msb = MultiStepBuffer(1000, 4)
        self.assertEqual(len(msb), 0)
        for i in range(1000):
            msb.push(i//4, np.array([1,1]), np.array([2,2]), np.array([3,3]), np.array([4,4]), np.array([5,5]), [0])
        self.assertEqual(len(msb), 1000//4 -1)
        x = msb.sample_batch(32)
        self.assertEqual(len(x), 6)
        self.assertEqual(x[0].shape, (32,4,2))

        y = msb.pop(2)
        y1 = msb.convert(y)
        self.assertEqual(len(y1), 6)
        self.assertEqual(y1[0].shape, (2,4,2))
        
        
        
        

if __name__ == '__main__':
    t = unittest.TestSuite()
    t.addTest(MultiStepBufferTest('test_msb_unit'))
    run=unittest.TextTestRunner()
    run.run(t)