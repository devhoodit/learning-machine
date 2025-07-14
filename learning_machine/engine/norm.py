import sklearn.preprocessing as sk_processing
from learning_machine.engine import DataEngine
from learning_machine.zoo import DATA_ENGINE_ZOO
import pandas as pd


@DATA_ENGINE_ZOO.regist()
class StandardScaler(DataEngine):
    def __init__(self, cols: list[str], return_new=False, prefix="standard_scale"):
        """Standard scaler from scikit-learn.

        Args:
            cols (list[str]): target columns
            return_new (bool, optional): return new scaled dataframe. If False, modify original dataframe. Defaults to False.
            prefix (str, optional): prefix of new dataframe. Valid when return_new is True. Defaults to "standard_scale".
        """
        self.cols = cols
        self.fit = False
        self.return_new = return_new
        self.prefix = prefix

        self.norm_engines = {}
        for col in cols:
            self.norm_engines[col] = sk_processing.StandardScaler()

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        scaled = {}
        if self.fit:
            for col, engine in self.norm_engines.items():
                scaled[col] = engine.transform(data[col].to_numpy().reshape(-1, 1))
            return data
        for col, engine in self.norm_engines.items():
            scaled[col] = engine.fit_transform(data[col].to_numpy().reshape(-1, 1))
        self.fit = True
        if self.return_new:
            dfdata = {f"{self.prefix}_{k}": v for k, v in scaled.items()}
            return pd.DataFrame(dfdata, index=data.index)

        for col, v in scaled.items():
            data[col] = v
        return data


@DATA_ENGINE_ZOO.regist()
class RobustScaler(DataEngine):
    def __init__(self, cols: list[str], return_new=False, prefix="robust_scale"):
        """RobustScaler from scikit-learn.

        Args:
            cols (list[str]): target columns
            return_new (bool, optional): return new scaled dataframe. If False, modify original dataframe. Defaults to False.
            prefix (str, optional): prefix of new dataframe. Valid when return_new is True. Defaults to "robust_scale".
        """
        self.cols = cols
        self.fit = False
        self.return_new = return_new
        self.prefix = prefix

        self.norm_engines = {}
        for col in cols:
            self.norm_engines[col] = sk_processing.RobustScaler()

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        scaled = {}
        if self.fit:
            for col, engine in self.norm_engines.items():
                scaled[col] = engine.transform(data[col].to_numpy().reshape(-1, 1))
            return data
        for col, engine in self.norm_engines.items():
            scaled[col] = engine.fit_transform(data[col].to_numpy().reshape(-1, 1))
        self.fit = True
        if self.return_new:
            dfdata = {f"{self.prefix}_{k}": v for k, v in scaled.items()}
            return pd.DataFrame(dfdata, index=data.index)

        for col, v in scaled.items():
            data[col] = v
        return data


@DATA_ENGINE_ZOO.regist()
class MinMaxScaler(DataEngine):
    def __init__(self, cols: list[str], return_new=False, prefix="min_max_scale"):
        """MinMaxScaler from scikit-learn.

        Args:
            cols (list[str]): _description_
            return_new (bool, optional): return new scaled dataframe. If False, modify original dataframe. Defaults to False.
            prefix (str, optional): prefix of new dataframe. Valid when return_new is True. Defaults to "min_max_scale".
        """
        self.cols = cols
        self.fit = False
        self.return_new = return_new
        self.prefix = prefix

        self.norm_engines = {}
        for col in cols:
            self.norm_engines[col] = sk_processing.MinMaxScaler()

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        scaled = {}
        if self.fit:
            for col, engine in self.norm_engines.items():
                data[col] = engine.transform(data[col].to_numpy().reshape(-1, 1))
            return data
        for col, engine in self.norm_engines.items():
            data[col] = engine.fit_transform(data[col].to_numpy().reshape(-1, 1))
        self.fit = True

        if self.return_new:
            dfdata = {f"{self.prefix}_{k}": v for k, v in scaled.items()}
            return pd.DataFrame(dfdata, index=data.index)

        for col, v in scaled.items():
            data[col] = v
        return data
