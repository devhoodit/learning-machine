from typing import Any
import numpy as np
import pandas as pd

from learning_machine.zoo import DATA_ENGINE_ZOO
from .engine import DataEngine


class NdFillSinkHole(DataEngine):
    """fill continuous nan value"""

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


@DATA_ENGINE_ZOO.regist("FillSinkHole")
class FillSinkHole(NdFillSinkHole):
    def __init__(self, col: str, length: int, fillwith: Any, name: str = ""):
        super().__init__(length, fillwith)
        self.column = col
        self.name = name
        if not name:
            self.name = col

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        arr = data[self.column].to_numpy()
        fill = super().__call__(arr)
        data[self.name] = fill
        return data


@DATA_ENGINE_ZOO.regist("DropNARow")
class DropNARow(DataEngine):
    "drop nan value row"

    def __init__(self, columns: list[str], copy=True):
        self.columns = columns
        self.copy = copy

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.copy:
            data = data.copy()

        return data.dropna(subset=self.columns)


@DATA_ENGINE_ZOO.regist("FillNa")
class FillNa(DataEngine):
    def __init__(self, cols: list[str], fillwith):
        self.cols = cols
        self.fillwith = fillwith

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        filler = {col: self.fillwith for col in self.cols}
        data.fillna(value=filler, inplace=True)
        return data


@DATA_ENGINE_ZOO.regist("FillNaFrom")
class FillNaFrom(DataEngine):
    def __init__(self, col: str, from_col: str):
        self.col = col
        self.from_col = from_col

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.col] = data[self.col].fillna(data[self.from_col], inplace=False)
        return data
