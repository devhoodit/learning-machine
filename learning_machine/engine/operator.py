import pandas as pd
import numpy as np

from .engine import DataEngine
from learning_machine.zoo import DATA_ENGINE_ZOO


@DATA_ENGINE_ZOO.regist()
class Add(DataEngine):
    def __init__(self, col1: str, col2: str, prefix: str = "add"):
        self.col1 = col1
        self.col2 = col2
        self.prefix = prefix

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        x1 = data[self.col1].to_numpy()
        x2 = data[self.col2].to_numpy()
        x3 = x1 + x2
        return pd.DataFrame(
            {
                f"{self.prefix}_{self.col1}_{self.col2}": x3,
            },
            index=data.index,
        )


@DATA_ENGINE_ZOO.regist()
class Sub(DataEngine):
    def __init__(self, col1: str, col2: str, prefix: str = "sub"):
        self.col1 = col1
        self.col2 = col2
        self.prefix = prefix

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        x1 = data[self.col1].to_numpy()
        x2 = data[self.col2].to_numpy()
        x3 = x1 - x2
        return pd.DataFrame(
            {
                f"{self.prefix}_{self.col1}_{self.col2}": x3,
            },
            index=data.index,
        )


@DATA_ENGINE_ZOO.regist()
class Mul(DataEngine):
    def __init__(self, col1: str, col2: str, prefix: str = "mul"):
        self.col1 = col1
        self.col2 = col2
        self.prefix = prefix

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        x1 = data[self.col1].to_numpy()
        x2 = data[self.col2].to_numpy()
        x3 = x1 * x2
        return pd.DataFrame(
            {
                f"{self.prefix}_{self.col1}_{self.col2}": x3,
            },
            index=data.index,
        )


@DATA_ENGINE_ZOO.regist()
class Div(DataEngine):
    def __init__(self, col1: str, col2: str, prefix: str = "div"):
        self.col1 = col1
        self.col2 = col2
        self.prefix = prefix

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        x1 = data[self.col1].to_numpy()
        x2 = data[self.col2].to_numpy()
        x3 = x1 / x2
        return pd.DataFrame(
            {
                f"{self.prefix}_{self.col1}_{self.col2}": x3,
            },
            index=data.index,
        )
