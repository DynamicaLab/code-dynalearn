from templates import *
from dynalearn.config import ExperimentConfig
from dynalearn.experiments import Experiment


class SamplerTest(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig.test(config="discrete")
        self.exp = Experiment(self.config, verbose=0)
        self.dataset = self.exp.dataset
        self.dataset.setup(self.exp)
        self.exp.generate_data()
        self.sampler = self.dataset.sampler

    def test_call(self):
        pass

    def test_update(self):
        pass

    def test_reset(self):
        pass

    def test_get_networks(self):
        pass

    def test_get_states(self):
        pass

    def tearDown(self):
        os.removedirs(f"./{self.exp.name}")


if __name__ == "__main__":
    unittest.main()
