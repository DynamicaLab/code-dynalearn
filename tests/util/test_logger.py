import os
import time
import unittest
import json

from dynalearn.util import Logger, LoggerDict


class LoggerTest(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()
        self.logger.log = {"something": 1, "something else": 2}
        self.log = {"new things": 2, "further new things": 5}
        self.filename = "./logger.json"

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_on_task_begin(self):
        pass

    def test_on_task_end(self):
        pass

    def test_on_task_update(self):
        pass

    def test_save(self):
        with open(self.filename, "w") as f:
            self.logger.save(f)
        with open(self.filename, "r") as f:
            log = json.load(f)
        self.assertEqual(log, self.logger.log)

    def test_load(self):
        with open(self.filename, "w") as f:
            json.dump(self.log, f, indent=4)
        with open(self.filename, "r") as f:
            self.logger.load(f)
        self.assertEqual(self.logger.log, self.log)
