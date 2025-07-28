import pandas as pd
import numpy as np
from typing import Literal

from .engine import DataEngine
from learning_machine.zoo import DATA_ENGINE_ZOO

operators = {
    "+": lambda x1, x2: x1 + x2,
    "-": lambda x1, x2: x1 - x2,
    "*": lambda x1, x2: x1 * x2,
    "/": lambda x1, x2: x1 / x2,
    "//": lambda x1, x2: x1 // x2,
    "%": lambda x1, x2: x1 % x2,
}


class BinaryOperator(DataEngine):
    """Binary operate with two columns. return {prefix}_{col1}_{col2} column dataframe"""

    def __init__(
        self,
        col1: str,
        col2: str,
        operator: Literal["+", "-", "*", "/", "//", "%"],
        prefix: str,
    ) -> None:
        """
        Args:
            col1 (str): _description_
            col2 (str): _description_
            operator (Literal["+", "-", "*", "/", "//", "%"]): binary operator
            prefix (str): prefix of result dataframe
        """
        super().__init__()
        self.col1 = col1
        self.col2 = col2
        self.operator = operator
        self.prefix = prefix

        if self.operator not in list(operators.keys()):
            raise ValueError(f"operator: {self.operator} if not defined")

        if not self.prefix:
            self.prefix = self.operator

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        augend = data[self.col1].to_numpy()
        addend = data[self.col2].to_numpy()

        result = operators[self.operator](augend, addend)
        return pd.DataFrame({f"{self.prefix}_{self.col1}_{self.col2}": result})


@DATA_ENGINE_ZOO.regist()
class Add(BinaryOperator):
    """Add two columns. return {prefix}_{col1}_{col2} column dataframe"""

    def __init__(self, col1: str, col2: str, prefix: str = "add"):
        """
        Args:
            col1 (str): augend column
            col2 (str): addend column
            prefix (str, optional): prefix of result dataframe. Defaults to "add".
        """
        super().__init__(col1, col2, "+", prefix)

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return super().__call__(data)


@DATA_ENGINE_ZOO.regist()
class Sub(BinaryOperator):
    """Subtract two columns. return {prefix}_{col1}_{col2} column dataframe"""

    def __init__(self, col1: str, col2: str, prefix: str = "sub"):
        """
        Args:
            col1 (str): augend column
            col2 (str): addend column
            prefix (str, optional): prefix of result dataframe. Defaults to "sub".
        """
        super().__init__(col1, col2, "-", prefix)
        self.prefix = prefix

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return super().__call__(data)


@DATA_ENGINE_ZOO.regist()
class Mul(BinaryOperator):
    """Multiply two columns. return {prefix}_{col1}_{col2} column dataframe"""

    def __init__(self, col1: str, col2: str, prefix: str = "mul"):
        """
        Args:
            col1 (str): augend column
            col2 (str): addend column
            prefix (str, optional): prefix of result dataframe. Defaults to "mul".
        """
        super().__init__(col1, col2, "*", prefix)

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        return super().__call__(data)


@DATA_ENGINE_ZOO.regist()
class Div(DataEngine):
    """Divide two columns. return {prefix}_{col1}_{col2}"""

    def __init__(self, col1: str, col2: str, prefix: str = "div"):
        """
        Args:
            col1 (str): augend column
            col2 (str): addend column
            prefix (str, optional): prefix of result dataframe. Defaults to "div".
        """
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
