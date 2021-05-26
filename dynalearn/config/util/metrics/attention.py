from dynalearn.config import Config


class AttentionConfig(Config):
    @classmethod
    def default(cls):
        cls = cls()
        cls.max_num_points = 100
        return cls
