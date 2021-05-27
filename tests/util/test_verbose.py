import os
import unittest

from dynalearn.util import Verbose


class VerboseClass(unittest.TestCase):
    def setUp(self):
        self.filename = "test_verbose.txt"
        self.vtypes = [0, 1, 2]

    def tearDown(self):
        os.remove(self.filename)

    def test_init(self):
        for vtype in self.vtypes:
            verbose = Verbose(self.filename, vtype=vtype)
