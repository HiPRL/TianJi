import sys
sys.path.insert(0, './../../../../')

from drl.utils.buffers import StepBuffer
import unittest
import numpy as np


class StepBufferTest(unittest.TestCase):
    def test_step_unit(self) -> None:
        buff = StepBuffer(100)
        self.assertEqual(len(buff), 0)
        for i in range(50):
            buff.push(i, 0, 0, 0, 0, 0, [0])
        self.assertEqual(len(buff), 50)

    def test_sample_batch(self) -> None:
        buff = StepBuffer(100)
        self.assertEqual(len(buff), 0)
        for i in range(121):
            buff.push(i, np.array([1,1]), np.array([2,2]), np.array([3,3]), np.array([4,4]), np.array([5,5]), [0])
        self.assertEqual(len(buff), 100)
        t = buff.sample_batch(20)
        self.assertEqual(len(t), 6)
        self.assertEqual(t[0].shape, (20,2))

        b = buff.pop(2)
        b1 = buff.convert(b)
        self.assertEqual(len(b1), 6)
        self.assertEqual(b1[0].shape, (2,2))
        



if __name__ == '__main__':
    t = unittest.TestSuite()
    t.addTest(StepBufferTest('test_step_unit'))
    t.addTest(StepBufferTest('test_sample_batch'))
    run=unittest.TextTestRunner()
    run.run(t)