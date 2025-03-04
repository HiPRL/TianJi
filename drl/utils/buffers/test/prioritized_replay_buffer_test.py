import sys
sys.path.insert(0, './../../../../')

from drl.utils.buffers import Experience
from drl.utils.buffers.prioritized_replay_buffer import MinSegmentTree, SumSegmentTree, PrioritizedReplayBuffer
import unittest
import numpy as np



class TestSegmentTree(unittest.TestCase):
    def test_tree_set(self):
        tree = SumSegmentTree(4)

        tree[2] = 1.0
        tree[3] = 3.0

        assert np.isclose(tree.sum(), 4.0)
        assert np.isclose(tree.sum(0, 2), 0.0)
        assert np.isclose(tree.sum(0, 3), 1.0)
        assert np.isclose(tree.sum(2, 3), 1.0)
        assert np.isclose(tree.sum(2, -1), 1.0)
        assert np.isclose(tree.sum(2, 4), 4.0)
        assert np.isclose(tree.sum(2), 4.0)

    def test_tree_set_overlap(self):
        tree = SumSegmentTree(4)

        tree[2] = 1.0
        tree[2] = 3.0

        assert np.isclose(tree.sum(), 3.0)
        assert np.isclose(tree.sum(2, 3), 3.0)
        assert np.isclose(tree.sum(2, -1), 3.0)
        assert np.isclose(tree.sum(2, 4), 3.0)
        assert np.isclose(tree.sum(2), 3.0)
        assert np.isclose(tree.sum(1, 2), 0.0)

    def test_prefixsum_idx(self):
        tree = SumSegmentTree(4)

        tree[2] = 1.0
        tree[3] = 3.0

        assert tree.find_prefixsum_idx(0.0) == 2
        assert tree.find_prefixsum_idx(0.5) == 2
        assert tree.find_prefixsum_idx(0.99) == 2
        assert tree.find_prefixsum_idx(1.01) == 3
        assert tree.find_prefixsum_idx(3.00) == 3
        assert tree.find_prefixsum_idx(4.00) == 3

    def test_prefixsum_idx2(self):
        tree = SumSegmentTree(4)

        tree[0] = 0.5
        tree[1] = 1.0
        tree[2] = 1.0
        tree[3] = 3.0

        assert tree.find_prefixsum_idx(0.00) == 0
        assert tree.find_prefixsum_idx(0.55) == 1
        assert tree.find_prefixsum_idx(0.99) == 1
        assert tree.find_prefixsum_idx(1.51) == 2
        assert tree.find_prefixsum_idx(3.00) == 3
        assert tree.find_prefixsum_idx(5.50) == 3

    def test_max_interval_tree(self):
        tree = MinSegmentTree(4)

        tree[0] = 1.0
        tree[2] = 0.5
        tree[3] = 3.0

        assert np.isclose(tree.min(), 0.5)
        assert np.isclose(tree.min(0, 2), 1.0)
        assert np.isclose(tree.min(0, 3), 0.5)
        assert np.isclose(tree.min(0, -1), 0.5)
        assert np.isclose(tree.min(2, 4), 0.5)
        assert np.isclose(tree.min(3, 4), 3.0)

        tree[2] = 0.7

        assert np.isclose(tree.min(), 0.7)
        assert np.isclose(tree.min(0, 2), 1.0)
        assert np.isclose(tree.min(0, 3), 0.7)
        assert np.isclose(tree.min(0, -1), 0.7)
        assert np.isclose(tree.min(2, 4), 0.7)
        assert np.isclose(tree.min(3, 4), 3.0)

        tree[2] = 4.0

        assert np.isclose(tree.min(), 1.0)
        assert np.isclose(tree.min(0, 2), 1.0)
        assert np.isclose(tree.min(0, 3), 1.0)
        assert np.isclose(tree.min(0, -1), 1.0)
        assert np.isclose(tree.min(2, 4), 3.0)
        assert np.isclose(tree.min(2, 3), 4.0)
        assert np.isclose(tree.min(2, -1), 4.0)
        assert np.isclose(tree.min(3, 4), 3.0)


class TestPrioritizedReplayBuffer(unittest.TestCase):
    def test_per_unit(self):
        per = PrioritizedReplayBuffer(100)
        self.assertEqual(len(per), 0)
        per.push(1, np.array([1,1]), np.array([2,2]), np.array([3,3]), np.array([4,4]), np.array([5,5]), [0])
        self.assertEqual(len(per), 1)
        per.push(2, np.array([2,2]), np.array([2,1]), np.array([3,2]), np.array([4,4]), np.array([5,5]), [1])
        self.assertEqual(len(per), 2)
        for i in range(48):
            per.push(i+3, np.array([2,2]), np.array([2,1]), np.array([3,2]), np.array([4,4]), np.array([5,5]), [1+i])
        self.assertEqual(len(per), 50)
        x = per.sample_batch(4)
        self.assertEqual(len(x), 6)
        self.assertEqual(x[0].shape, (4,2))




if __name__ == "__main__":
    t = unittest.TestSuite()
    t.addTest(TestSegmentTree('test_tree_set'))
    t.addTest(TestSegmentTree('test_tree_set_overlap'))
    t.addTest(TestSegmentTree('test_prefixsum_idx'))
    t.addTest(TestSegmentTree('test_prefixsum_idx2'))
    t.addTest(TestSegmentTree('test_max_interval_tree'))
    
    t.addTest(TestPrioritizedReplayBuffer('test_per_unit'))
    # t.addTest(TestPrioritizedReplayBuffer('test_max_interval_tree'))

    run=unittest.TextTestRunner()
    run.run(t)