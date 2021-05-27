import numpy as np
from itertools import product
from .config import Config
from .util import (
    AttentionConfig,
    ForecastConfig,
    LTPConfig,
    PredictionConfig,
    StationaryConfig,
    StatisticsConfig,
)


class MetricsConfig(Config):
    @classmethod
    def test(cls):
        cls = cls()
        cls.names = []
        cls.ltp = LTPConfig.default()
        cls.prediction = PredictionConfig.default()
        cls.statistics = StatisticsConfig.default()
        cls.stationary = StationaryConfig.test()
        cls.forecast = ForecastConfig.default()
        cls.attention = AttentionConfig.default()

        return cls

    @classmethod
    def sis(cls):
        cls = cls()
        cls.names = []
        cls.ltp = LTPConfig.default()
        cls.prediciton = PredictionConfig.default()
        cls.statistics = StatisticsConfig.default()
        cls.stationary = StationaryConfig.sis()
        cls.attention = AttentionConfig.default()

        return cls

    @classmethod
    def plancksis(cls):
        cls = cls()
        cls.names = []
        cls.ltp = LTPConfig.default()
        cls.prediciton = PredictionConfig.default()
        cls.statistics = StatisticsConfig.default()
        cls.stationary = StationaryConfig.plancksis()
        cls.attention = AttentionConfig.default()

        return cls

    @classmethod
    def sissis(cls):
        cls = cls()
        cls.names = []
        cls.ltp = LTPConfig.default()
        cls.prediciton = PredictionConfig.default()
        cls.statistics = StatisticsConfig.default()
        cls.stationary = StationaryConfig.sissis()
        cls.attention = AttentionConfig.default()

        return cls

    @classmethod
    def dsir(cls):
        cls = cls()
        cls.names = []
        cls.prediction = PredictionConfig.default()
        cls.forecast = ForecastConfig.default()
        cls.stationary = StationaryConfig.dsir()
        cls.attention = AttentionConfig.default()

        return cls

    @classmethod
    def covid(cls):
        cls = cls()
        cls.names = []
        cls.prediction = PredictionConfig.default()
        cls.forecast = ForecastConfig.default()
        cls.attention = AttentionConfig.default()

        return cls
