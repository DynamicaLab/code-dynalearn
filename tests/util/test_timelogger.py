import unittest
import time

from datetime import datetime
from dynalearn.util import TimeLogger


class TimeLoggerTest(unittest.TestCase):
    def setUp(self):
        self.logger = TimeLogger()
        self.num_updates = 10

    def test_on_task_begin(self):
        self.logger.on_task_begin()
        self.assertTrue(self.logger.begin is not None)
        self.assertTrue(isinstance(self.logger.begin, datetime))
        self.assertTrue(self.logger.end is None)
        for k in ["begin"]:
            self.assertTrue(k in self.logger.log)

    def test_on_task_end(self):
        self.logger.on_task_begin()
        self.logger.on_task_end()
        self.assertTrue(self.logger.begin is not None)
        self.assertTrue(isinstance(self.logger.begin, datetime))
        self.assertTrue(self.logger.end is not None)
        self.assertTrue(isinstance(self.logger.end, datetime))
        for k in ["begin", "end", "time", "total"]:
            self.assertTrue(k in self.logger.log)

    def test_on_task_update(self):
        self.logger.on_task_begin()
        self.assertTrue(self.logger.begin is not None)
        self.assertTrue(isinstance(self.logger.begin, datetime))
        for i in range(self.num_updates):
            self.logger.on_task_update()
            self.assertTrue(self.logger.update is not None)
        self.logger.on_task_end()
        self.assertTrue(self.logger.end is not None)
        self.assertTrue(isinstance(self.logger.end, datetime))
        for k in ["begin", "end", "time", "total", "time-update"]:
            self.assertTrue(k in self.logger.log)

    def test_format_diff(self):
        t0 = datetime.now()
        time.sleep(1)
        t1 = datetime.now()
        self.assertEqual(self.logger.format_diff(t0, t1, to_sec=True), 1)
        d = self.logger.format_diff(t0, t1, to_sec=False)
        self.assertEqual(len(d), 4)
        self.assertEqual(d[0], 0)
        self.assertEqual(d[1], 0)
        self.assertEqual(d[2], 0)
        self.assertEqual(d[3], 1)
