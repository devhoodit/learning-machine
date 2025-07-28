import numpy as np
import pandas as pd
from learning_machine.zoo import DATA_ENGINE_ZOO
from .engine import DataEngine
from .engine_type import DataEngineType


def sin_cos_transform(x: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    phase = 2 * np.pi * x / period
    return np.sin(phase), np.cos(phase)


@DATA_ENGINE_ZOO.regist()
class SinCos(DataEngine):
    engine_type = [DataEngineType.RETURN_NEW_PD]
    """Periodically interpret the data"""

    def __init__(self, col: str, prefix="", period: float = 1.0):
        """
        Args:
            col (str):
            prefix (str, optional): prefix of return dataframe column name. Defaults to "".
            period (float, optional): period of the data. Defaults to 1.0.
        """
        self.col = col
        self.prefix = prefix
        self.period = period

        if not prefix:
            self.prefix = col

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        x = data[self.col].to_numpy().astype(np.float64)
        sinx, cosx = sin_cos_transform(x, self.period)
        return pd.DataFrame(
            {
                f"{self.prefix}_sin": sinx,
                f"{self.prefix}_cos": cosx,
            },
            index=data.index,
        )
