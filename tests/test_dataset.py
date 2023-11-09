import unittest
import numpy as np

from src import dataset


class TestDataset(unittest.TestCase):
    def test_next_simple(self):
        xs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ys = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        d = dataset.Dataset(xs, ys)
        for i in range(10):
            x, y = next(d)
            self.assertNotIn(x, [7, 8, 9, 10])
            self.assertNotEqual(y, 2)
