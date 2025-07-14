import sklearn.preprocessing as skp
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from .engine import DataEngine
from learning_machine.zoo import DATA_ENGINE_ZOO


@DATA_ENGINE_ZOO.regist()
class OneHotEncoder(DataEngine):
    def __init__(self, cols: list[str], prefix: str = "onehot", sparse_output=False):
        """onehot encoder from scikit-learn. return columns {prefix}_{col}.

        Args:
            cols (list[str]): list of columns need encoding
            prefix (str, optional): return column prefix. Defaults to "onehot".
            sparse_output (bool, optional): return sparse matrix. Ref scikit-learn. Defaults to False.
        """
        self.cols = cols
        self.prefix = prefix
        self.enc = skp.OneHotEncoder(sparse_output=sparse_output)
        self.is_fit = False

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        arr = data[self.cols].to_numpy()
        if not self.is_fit:
            one_hot = self.enc.fit_transform(arr)
            self.is_fit = True
        else:
            one_hot = self.enc.transform(arr)  # type: ignore
            if isinstance(one_hot, csr_matrix):
                one_hot = one_hot.toarray()

        col_names = np.concatenate(self.enc.categories_)
        col_names = [f"{self.prefix}_{col}" for col in col_names]

        return pd.DataFrame(one_hot, columns=col_names, index=data.index)  # type: ignore


@DATA_ENGINE_ZOO.regist()
class LabelEncoder(DataEngine):
    def __init__(self, col: str, prefix: str = "label"):
        """label encoder from scikit-learn. return column {prefix}_{col}.

        Args:
            col (str): column name
            prefix (str, optional): return column prefix. Defaults to "label".
        """
        self.col = col
        self.prefix = prefix
        self.enc = skp.LabelEncoder()
        self.is_fit = False

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        arr = data[self.col].to_numpy()
        if not self.is_fit:
            label = self.enc.fit_transform(arr)
            self.is_fit = True
        else:
            label = self.enc.transform(arr)

        return pd.DataFrame({f"{self.prefix}_{self.col}": label})
