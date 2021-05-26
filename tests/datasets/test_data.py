import numpy as np
import warnings

warnings
from unittest import TestCase
from dynalearn.datasets import DataCollection, StateData


class TestData(TestCase):
    def setUp(self):
        self.name = "data"
        self.size = 10
        self.shape = (5, 6)
        self.num_points = 14
        self.data = DataCollection(
            name=self.name,
            data_list=[
                StateData(name=self.name, data=np.random.rand(self.size, *self.shape))
                for i in range(self.num_points)
            ],
        )

    def test_get(self):
        index = np.random.randint(self.num_points)
        self.assertEqual(self.data[index].shape, self.shape)

    def test_size(self):
        index = np.random.randint(self.num_points)
        self.assertEqual(self.data[index].size, self.size)

    def test_num_points(self):
        self.assertEqual(self.data.size, self.num_points)


if __name__ == "__main__":
    unittest.main()
