import unittest

from dynalearn.config import Config, DynamicsConfig


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.dynamics = DynamicsConfig.sis()
        self.config = Config()
        self.config.dynamics = self.dynamics
        self.config.seed = 0

    def test_init(self):
        self.config

    def test_getitem(self):
        self.assertEqual(
            self.config["dynamics/infection"], self.config.dynamics.infection
        )
        self.assertEqual(self.config["seed"], 0)

    def test_haslist(self):
        config_with_list = self.config.copy()
        config_with_list.dynamics.alpha = [0.0, 0.5, 1.0]
        self.assertTrue(config_with_list.has_list())
        self.assertFalse(self.config.has_list())

    def test_statedict(self):
        self.assertEqual(self.config.state_dict["dynamics/name"], "SIS")
        self.assertEqual(self.config.state_dict["dynamics/infection"], 0.04)
        self.assertEqual(self.config.state_dict["dynamics/recovery"], 0.08)
        self.assertEqual(self.config.state_dict["dynamics/init_param"], None)
        self.assertEqual(self.config.state_dict["seed"], 0)


if __name__ == "__main__":
    unittest.main()
