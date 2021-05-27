from templates import *
from dynalearn.nn import History


class HistoryTest(unittest.TestCase):
    def setUp(self):
        self.history = History()
        return

    def test_update_log(self):
        logs = {"time": 0, "loss": 1, "metrics": 2}
        self.history.update_epoch(logs)
        self.assertEqual({0: logs}, self.history._epoch_logs)
        self.history.update_epoch(logs)
        self.assertEqual({0: logs, 1: logs}, self.history._epoch_logs)

        self.history.update_batch(logs)
        self.assertEqual({0: logs}, self.history._batch_logs)
        self.history.update_batch(logs)
        self.assertEqual({0: logs, 1: logs}, self.history._batch_logs)
        self.history.reset()


if __name__ == "__main__":
    unittest.main()
