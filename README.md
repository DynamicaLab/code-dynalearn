# code-dynalearn
Deep learning of contagion dynamics on complex networks in pytorch

[![DOI](https://zenodo.org/badge/371061340.svg)](https://zenodo.org/badge/latestdoi/371061340)

## Requirements and dependencies

* `torch>=1.6.0`
* `torch_geometric>=1.6.3`

## Installation
First, clone this repository.
```bash
git clone https://github.com/DynamicaLab/code-dynalearn
```
Second, use pip to install the module.
```bash
pip install ./code-dynalearn
```

## Publications

Please cite:

_Deep learning of contagion dynamics on complex networks_<br>
[Charles Murphy](https://scholar.google.ca/citations?user=xgBmSD8AAAAJ&hl=en&oi=sra),
[Edward Laurence](https://edwardlaurence.me/) and
[Antoine Allard](http://antoineallard.info),<br>
[Nature Communications 12, 4720 (2021)](https://doi.org/10.1038/s41467-021-24732-2)

## How to use the code
This Python module defines different classes for the purpose of learning dynamics on networks, such as `Experiment`, `Config`, `Metrics`, `Dynamics`, `Model`, `Network` and `Dataset`. We review these classes and show how to use them in this section.


### Reproduce the data from the paper
The scripts to re-run all the experiments presented in the papers are [here](./scripts/launchers). To run a script, for instance the script [run-synthetic-discrete.py](./scripts/launchers/run-synthetic-discrete.py) that generates the data for Figs. 2, 3 and 4 for the simple, complex and interaction dynamics, simply run the following command:
```bash
python run-synthetic-discrete.py
```

To remake the figures, we refer to these [notebooks](./notebooks/figures).

### Building blocks
The building blocks of this modules include classes for dynamical models (`Dynamics`), neural network models (`Model`), graph / network models (`Network`) and dataset managers (`Dataset`).


#### `Dynamics` class
The `Dynamics` class is a virtual class, which can only be used to define subclasses. To define a subclass of `Dynamics`, one needs to define the `inital_state(self)` method, which returns an initial state, the `predict(self, x)` method, which computes the local transition probabilities of each node into a `np.array` given the current state x, the `loglikelihood(self, x)` method, which computes the log-probability of generating the time series _x_, the `sample(self, x)` method, which samples a next state given _x_ using the transition probabilities, and the `is_dead(self, x)` method, which determines if the state _x_ has reached a fixed point.

To define a `Dynamics` subclass object, such as a `SIS` object for example, one can procede in this way
```python
config = dynalearn.config.DynamicsConfig.sis()
dynamics = dynalearn.dynamics.SIS(config=config)
dynamics = dynalearn.dynamics.SIS(infection=0.5, recovery=0.5)
```

To run the dynamics a specific network, simply pass a `Network` object to it:
```python
dynamics.network = dynalearn.networks.Network(data=nx.gnp_random_graph(100, 0.1))
x0 = dynamics.initial_state()
x = [x0]
for t in range(10):
    x.append(dynamics.sample(x[-1]))
```

The graph neural network models are also defined as `Dynamics`, as an overhead of the `Model` class.

To construct a `Dynamics` objects, one can either give the parameters of the model in keywords, or pass a `DynamicsConfig` object. If some parameters are not defined, an error will be raised.

#### `Network` class
The `Network` class is an overhead of the `nx.Graph` class in _networkx_. It contains a `nx.Graph` in the attribute `data`, and defines and computes automatically the attributes as needed, such as node / edge attributes, and the edge list. To define a `Network` object, use
```python
g = dynalearn.networks.Network(data=nx.Graph())
```

The `NetworkGenerator` class generates network using the method `generate(self, seed=None)` and returns `Network` object.

The `WeightGenerator` class generates edge weights for a network by using the magic method `__call__(self, g)`.

The `NetworkTransform` class applies a transformation (e.g., removing a fraction _p_ of the edges randomly) to a network using the magic method `__call__(self, g)`.

To construct any of these objects, one can either give the parameters of the model in keywords, or pass a `NetworkConfig` object. One can also construct easily a `NetworkGenerator` object using the `dynalearn.networks.getter.get(config)` function, which also takes a `NetworkConfig` object.


#### `Model` class
The `Model` class is subclass of the `nn.Module` class in _pytorch_, which is also virtual. To define a subclass of `Model`, one needs to define the `forward(self, x, net_attr)` method, which takes a state _x_ and a network attributes tuple _net_attr_ containing the node / edge attributes and the edge list, and the `loss(self, y_true, y_pred)` method. A `Model` subclass works similarly to how _keras_ models work, i.e., where a `fit` method is used to train the model. A model can also save / load its weights easily.

We define a general `GraphNeuralNetwork` class that inherits from the `Model` class, which is also an overhead for the _torch_geometric_ layers and the `DynamicsGATConv` layer designed in the paper.


#### `Dataset` class
The `Dataset` class manages the dynamics and network models in order to generate datasets and split it into training, validation and / or test datasets. It also manages a `Sampler` and a `Weight` objects for the importance sampling procedure used in the paper. To define a `Dataset` subclass, one must pass `DatasetConfig` object:
```python
config = dynalearn.config.DiscreteDatasetConfig.plain()
dataset = dynalearn.dataset.DiscreteDataset(config)
```

To generate some data and split it, one needs to pass in inputs the `Experiment` object as follows:
```python
dataset.generate(exp)
val_dataset = dataset.partition(type="random", fraction=0.5)
```

This generates another `Dataset` subclass object with half of the data contained in `dataset` selected at random.


### Script and experimentation tools
We define a set of experimentation tools using these building blocks, for easy written, readable and versatile scripts, based on the `Experiment`, `Config` and `Metrics` classes. The example of experiment is presented in the [notebook example](./notebooks/example-sis-ba.ipynb).


#### `Experiment` class
The role of an `Experiment` object is to manage all the other objects, that is, to generate a dataset, to train a model, to compute some postprocessing metrics, etc.. Therefore this object contains all the information needed to perform an experiment including all the parameters of the different models, but also the paths where the data generated within the experiment can be recovered.

To define an `Experiment` object, one passes a `Config` object (see below) in input to the experiment as follows:
```python
config = dynalearn.config.ExperimentConfig.default("exp-name", "sis", "gnp")
exp = dynalearn.experiments.Experiment(config, verbose=0)
```

At this point, the `exp` object can be used to run a complete experiment, by running the `run` method:
```python
exp.run(tasks=None)
```

The `run` method takes in input the set of tasks one wants to perform, represented as a `str`, a list of `str` or `None` (where it runs all the tasks). All the available tasks are contains in the `__tasks__` attribute of `exp`, which include data generation, training, saving, loading, etc..


#### `Config` class
A `Config` object contains all the information needed to define an `Experiment` object. It can also be used to define `Dynamics`, `Networks` and `Dataset` objects, among others.

To define a `Config` object, procede in this way:
```python
config = dynalearn.config.Config()
```

Then, one can add any attributes to the `Config` object. For instance, assume that we want to define a `Config` object for a random graph, such as G(N, p). Then, we might want the `Config` to contain _N_, the number of nodes, _p_, the connection probability, and the name of the network model:
```python
network_config = Config()
network_config.name = "G(N, p)"
network_config.N = 100
network_config.p = 0.1
```

A `Config` object can be displayed using `print(network_config)`, and can be represented by a dictionary with the attribute `network_config.state_dict`. It is also possible to give another `Config` object as a parameter of `network_config`. In the `state_dict` attribute, the `Config` hierarchy is encoded using `/`. This is an effective way of defining a `Config` object for an `Experiment`.

It is recommended that the parameters of a `Config` object are picklable, as it is convenient to save it using `pickle`.

We define different `Config` subclasses for each building block class, such as `NetworkConfig`, `DynamicsConfig`, `DatasetConfig`, among others, which include different _classmethods_.


#### `Metrics` class
Within an `Experiment` object, we define `Metrics` objects that can be computed, for instance, after training a model. The `Metrics` class is virtual, and can be used only in the context of defining subclasses of it. A subclass of `Metrics` must define the method `compute(self, exp)`, which takes an `Experiment` object in input and, as the name intends, computes the values of the metrics for this experiment. A `Metrics` subclass can also save / load its data to a _HDF5_ file.
