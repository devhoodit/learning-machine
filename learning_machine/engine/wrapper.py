from __future__ import annotations

import pandas as pd
from .engine import DataEngine


def wrap_df2nd(engine: DataEngine, columns: str | list[str], copy=True) -> DF2NDarr:
    return DF2NDarr(engine, columns, copy)


class DF2NDarr(DataEngine):
    def __init__(self, engine: DataEngine, columns: str | list[str], copy=True):
        self.engine = engine
        self.copy = copy
        self._type = "str"
        if isinstance(columns, str):
            self._type = "str"
            self.columns = [columns]
        elif isinstance(columns, list):
            self._type = "list"
            self.columns = columns
        else:
            raise ValueError(f"unexpected input type. expect str or list[str] but {type(columns)}")

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.copy:
            data = data.copy()

        for col in self.columns:
            col_type = data[col].dtype
            arr = data[col].to_numpy()
            arr = self.engine(arr)
            data[col] = arr
            data[col] = data[col].astype(col_type)

        return data
