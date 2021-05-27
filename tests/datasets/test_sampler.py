import os

from shutil import rmtree
from .templates import *
from dynalearn.config import ExperimentConfig
from dynalearn.experiments import Experiment

NUM_SAMPLES = 10


class SamplerTest(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig.test(config="discrete")
        self.exp = Experiment(self.config, verbose=0)
        self.dataset = self.exp.dataset
        self.dataset.setup(self.exp)
        self.exp.generate_data()
        self.sampler = self.dataset.sampler

    def tearDown(self):
        if os.path.exists("./test"):
            rmtree("./test")

    def test_call(self):
        for i in range(NUM_SAMPLES):
            out = self.sampler()
            self.assertTrue(isinstance(out, tuple))
            self.assertEqual(len(out), 2)
            self.assertTrue(out[0] < self.exp.train_details.num_networks)
            self.assertTrue(out[1] < self.exp.train_details.num_samples)

    def template_update(self, replace=True):
        self.sampler.replace = replace
        networks = self.sampler.avail_networks.copy()
        states = {k: v.copy() for k, v in self.sampler.avail_states.items()}
        i = 0
        for g in networks:
            for s in states[g]:
                self.assertEqual(i, self.sampler.counter)
                self.sampler.update(g, s)
                if replace:
                    self.assertTrue(s in self.sampler.avail_states[g])
                else:
                    self.assertFalse(s in self.sampler.avail_states[g])
                i += 1
            if replace:
                self.assertTrue(g in self.sampler.avail_networks)
            else:
                self.assertFalse(g in self.sampler.avail_networks)

    def test_update_with_replace(self):
        self.template_update(replace=True)

    def test_update_without_replace(self):
        self.template_update(replace=False)

    def test_reset(self):
        self.sampler.replace = False
        networks = self.sampler.avail_networks.copy()
        states = {k: v.copy() for k, v in self.sampler.avail_states.items()}
        for i in range(NUM_SAMPLES):
            self.sampler()
        self.assertNotEqual(networks, self.sampler.avail_networks)
        self.assertNotEqual(states, self.sampler.avail_states)
        self.assertEqual(self.sampler.counter, NUM_SAMPLES)
        self.sampler.reset()
        self.assertEqual(self.sampler.counter, 0)
        self.assertEqual(networks, self.sampler.avail_networks)
        self.assertEqual(states, self.sampler.avail_states)


if __name__ == "__main__":
    unittest.main()
