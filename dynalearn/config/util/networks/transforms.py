from dynalearn.config import Config


class TransformConfig(Config):
    @classmethod
    def sparcifier(cls):
        cls = cls()
        cls.names = ["SparcifierTransform"]
        cls.maxiter = 100
        cls.p = -1
        return cls
