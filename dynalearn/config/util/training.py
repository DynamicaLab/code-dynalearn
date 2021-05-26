from dynalearn.config import Config


class TrainingConfig(Config):
    @classmethod
    def default(
        cls,
    ):
        cls = cls()

        cls.val_fraction = 0.01
        cls.val_bias = 0.8
        cls.epochs = 30
        cls.batch_size = 32
        cls.num_nodes = 1000
        cls.num_networks = 1
        cls.num_samples = 10000
        cls.resampling = 2
        cls.maxlag = 1
        cls.resample_when_dead = True

        return cls

    @classmethod
    def discrete(
        cls,
    ):
        cls = cls()

        cls.val_fraction = 0.01
        cls.val_bias = 0.8
        cls.epochs = 30
        cls.batch_size = 1
        cls.num_networks = 1
        cls.num_samples = 10000
        cls.resampling = 2
        cls.maxlag = 1
        cls.resample_when_dead = True

        return cls

    @classmethod
    def continuous(
        cls,
    ):
        cls = cls()

        cls.val_fraction = 0.1
        cls.val_bias = 0.5
        cls.epochs = 30
        cls.batch_size = 1
        cls.num_networks = 1
        cls.num_samples = 10000
        cls.resampling = 100
        cls.maxlag = 1
        cls.resample_when_dead = False

        return cls

    @classmethod
    def test(
        cls,
    ):
        cls = cls()

        cls.val_fraction = 0.01
        cls.val_bias = 0.8
        cls.epochs = 5
        cls.batch_size = 10
        cls.num_networks = 1
        cls.num_samples = 10
        cls.resampling = 2
        cls.maxlag = 1
        cls.resample_when_dead = True

        return cls
