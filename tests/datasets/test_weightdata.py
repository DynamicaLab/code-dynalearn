import unittest

from templates import DiscreteWeightTest, ContinuousWeightTest
from dynalearn.datasets.weights import (
    Weight,
    DegreeWeight,
    StrengthWeight,
    DiscreteStateWeight,
    DiscreteCompoundStateWeight,
    ContinuousStateWeight,
    ContinuousCompoundStateWeight,
    StrengthContinuousStateWeight,
    StrengthContinuousCompoundStateWeight,
)


class DWeightTest(DiscreteWeightTest, unittest.TestCase):
    def get_weight(self):
        return Weight()


class DDegreeWeightTest(DiscreteWeightTest, unittest.TestCase):
    def get_weight(self):
        return DegreeWeight()


class DStrengthWeightTest(DiscreteWeightTest, unittest.TestCase):
    def get_weight(self):
        return StrengthWeight()


class DiscreteStateWeightTest(DiscreteWeightTest, unittest.TestCase):
    def get_weight(self):
        return DiscreteStateWeight()


class DiscreteCompoundStateWeightTest(DiscreteWeightTest, unittest.TestCase):
    def get_weight(self):
        return DiscreteCompoundStateWeight()


class CWeightTest(ContinuousWeightTest, unittest.TestCase):
    def get_weight(self):
        return Weight()


class CDegreeWeightTest(ContinuousWeightTest, unittest.TestCase):
    def get_weight(self):
        return DegreeWeight()


class CStrengthWeightTest(ContinuousWeightTest, unittest.TestCase):
    def get_weight(self):
        return StrengthWeight()


class ContinuousStateWeightTest(ContinuousWeightTest, unittest.TestCase):
    def get_weight(self):
        return ContinuousStateWeight()


class ContinuousCompoundStateWeightTest(ContinuousWeightTest, unittest.TestCase):
    def get_weight(self):
        return ContinuousCompoundStateWeight()


class StrengthContinuousStateWeightTest(ContinuousWeightTest, unittest.TestCase):
    def get_weight(self):
        return StrengthContinuousStateWeight()


class StrengthContinuousCompoundStateWeightTest(
    ContinuousWeightTest, unittest.TestCase
):
    def get_weight(self):
        return StrengthContinuousCompoundStateWeight()


if __name__ == "__main__":
    unittest.main()
