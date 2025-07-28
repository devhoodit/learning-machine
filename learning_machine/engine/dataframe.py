from __future__ import annotations
import pandas as pd
from learning_machine.engine import DataEngine, create_engines_from_config
from learning_machine.zoo.zoo import DATA_ENGINE_ZOO


@DATA_ENGINE_ZOO.regist()
class ConcatDFs(DataEngine):
    """Concat the outputs of the engines into pd.Dataframe"""

    def __init__(self, engines: list[DataEngine]):
        """
        Args:
            engines (list[DataEngine]): engines
        """
        super().__init__()
        self.engines = engines

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        datas = []
        for engine in self.engines:
            datas.append(engine(data))
        return pd.concat([data] + datas, axis=1)

    @classmethod
    def from_config(cls, config: list) -> ConcatDFs:
        engines = create_engines_from_config(config)
        return cls(engines)


@DATA_ENGINE_ZOO.regist()
class DropColumns(DataEngine):
    """Drop columns from dataframe"""

    def __init__(self, cols: list[str], copy=True):
        """
        Args:
            cols (list[str]): columns that want to drop
            copy (bool, optional): copy data and return new data that processed. If False, it can effect to original data. Defaults to True.
        """
        self.drop_cols = cols
        self.copy = copy

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.copy:
            data = data.copy()

        for col in self.drop_cols:
            data.pop(col)

        return data
