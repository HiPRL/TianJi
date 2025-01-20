import sys
sys.path.insert(0, './../../../../')

from drl.utils.buffers import EpisodeBuffer
import unittest
import numpy as np




class EpisodeBufferTest(unittest.TestCase):
    def test_episode_unit(self) -> None:
        buff = EpisodeBuffer(10, 10)
        self.assertEqual(len(buff), 0)
        self.assertFalse(buff.push_flag)
        for i in range(0, 51):
            buff.push(i//10, i, 0, 0, 0, 0, [0])
        self.assertEqual(len(buff), 5)

    def test_sample_batch(self) -> None:
        buff = EpisodeBuffer(10, 12)
        self.assertEqual(len(buff), 0)
        for i in range(12 * 9 + 1):
            buff.push(i//12, 0, 0, 0, 0, 0, [0])
        self.assertEqual(len(buff), 9)
        # breakpoint()
        tmp = buff.sample_batch(3)
        self.assertEqual(len(tmp), 6)
        self.assertEqual(tmp[0].shape, (3,12))
        self.assertEqual(tmp[1].shape, (3,12))
        self.assertEqual(len(tmp[-1]), 1)
        self.assertEqual(tmp[-1][0].shape, (3,12))
        single_sample = buff.sample_batch(1)
        buff.push(single_sample,force=True)
        self.assertEqual(len(buff), 10)

        


if __name__ == '__main__':
    t = unittest.TestSuite()
    t.addTest(EpisodeBufferTest('test_episode_unit'))
    t.addTest(EpisodeBufferTest('test_sample_batch'))
    run=unittest.TextTestRunner()
    run.run(t)