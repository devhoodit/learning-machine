from typing import Any
import numpy as np
import pandas as pd

from learning_machine.zoo import DATA_ENGINE_ZOO
from .engine import DataEngine


class NdFillSinkHole(DataEngine):
    """Fill continuous nan value."""

    def __init__(self, length: int, fillwith: Any):
        self.length = length
        self.fillwith = fillwith

    def __call__(self, data: np.ndarray) -> np.ndarray:
        arr = data.copy()

        not_nan_mask = ~np.isnan(arr)
        not_nan_mask = np.concatenate(([True], not_nan_mask, [True]))
        range_arr = np.flatnonzero(not_nan_mask[1:] != not_nan_mask[:-1]).reshape(-1, 2)

        fill_range = range_arr[range_arr[:, 1] - range_arr[:, 0] < self.length]
        concat_arr = [np.arange(x[0], x[1]) for x in fill_range]
        if len(concat_arr) == 0:
            return arr
        idxs = np.concatenate(concat_arr)
        arr[idxs] = self.fillwith
        return arr


@DATA_ENGINE_ZOO.regist()
class FillSinkHole(NdFillSinkHole):
    """Fill in the interval where nan value appear continuously with specific value."""

    def __init__(self, col: str, length: int, fillwith: Any):
        """
        Args:
            col (str): target column
            length (int): interval length
            fillwith (Any): fill value
            name (str, optional): . Defaults to "".
        """
        super().__init__(length, fillwith)
        self.col = col

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        arr = data[self.col].to_numpy()
        fill = super().__call__(arr)
        data[self.col] = fill
        return data


@DATA_ENGINE_ZOO.regist()
class DropNARow(DataEngine):
    """Drop rows contain missing value in specific columns."""

    def __init__(self, cols: list[str], copy=True):
        """
        Args:
            cols (list[str]): columns to drop nan values
            copy (bool, optional): copy new dataframe and process. Defaults to True.
        """
        self.columns = cols
        self.copy = copy

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.copy:
            data = data.copy()

        return data.dropna(subset=self.columns)


@DATA_ENGINE_ZOO.regist()
class FillNaWithValue(DataEngine):
    """Fill nan rows with specific value."""

    def __init__(self, cols: list[str], fillwith):
        """
        Args:
            cols (list[str]): columns to fill
            fillwith (_type_): fill value
        """
        self.cols = cols
        self.fillwith = fillwith

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        filler = {col: self.fillwith for col in self.cols}
        data.fillna(value=filler, inplace=True)
        return data


@DATA_ENGINE_ZOO.regist()
class FillNaFrom(DataEngine):
    """Fill nan value from another column."""

    def __init__(self, col: str, from_col: str):
        """
        Args:
            col (str): target column
            from_col (str): from column
        """
        self.col = col
        self.from_col = from_col

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.col] = data[self.col].fillna(data[self.from_col], inplace=False)
        return data
