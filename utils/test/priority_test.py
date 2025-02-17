# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, './../..') 

import unittest
from utils import get_priority, Priority


class PriorityTest(unittest.TestCase):
    def test_get_priority_enum(self):
        """test returns"""
        self.assertEqual(get_priority(Priority.HIGHEST), 0)
        self.assertEqual(get_priority(Priority.VERY_HIGH), 10)
        self.assertEqual(get_priority(Priority.LOWEST), 100)
    
    def test_get_priority_value(self):
        """test passing a vaild or invaild value"""
        self.assertEqual(get_priority(0), 0)
        self.assertEqual(get_priority(50), 50)
        self.assertEqual(get_priority("highest"), 0)
        with self.assertRaises(ValueError):
            get_priority(-1)
        with self.assertRaises(ValueError):
            get_priority(101)
        with self.assertRaises(KeyError):
            get_priority("unkown")
        with self.assertRaises(TypeError):
            get_priority(101.1)

        


if __name__ == '__main__':
    t = unittest.TestSuite()
    t.addTest(PriorityTest('test_get_priority_enum'))
    t.addTest(PriorityTest('test_get_priority_value'))
    run=unittest.TextTestRunner()
    run.run(t)