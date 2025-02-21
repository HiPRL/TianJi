# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, './../..')  

import unittest
import numpy as np
from utils import ScalarBuffer


class TestScalarBuffer(unittest.TestCase):
    def test_update_and_data(self):
        buf = ScalarBuffer()
        buf.update({'loss': 0.1, 'accuracy': 0.9}, index=1)
        
        results = list(buf.data())
        expected = {
            'loss': (1, 0.1),
            'accuracy': (1, 0.9)
        }
        
        for key, value_arr, idx in results:
            self.assertEqual(idx, expected[key][0])
            self.assertAlmostEqual(value_arr.item(), expected[key][1])
    
    def test_clear(self):
        buf = ScalarBuffer()
        buf.update({'loss': 0.1}, index=1)
        self.assertTrue(buf.buffer) 
        buf.clear()
        self.assertEqual(len(buf.buffer), 0)  


if __name__ == '__main__':
    t = unittest.TestSuite()
    t.addTest(TestScalarBuffer('test_update_and_data'))
    t.addTest(TestScalarBuffer('test_clear'))
    r = unittest.TextTestRunner()
    r.run(t)