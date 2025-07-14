import pandas as pd
import numpy as np

from .engine import DataEngine
from .engine_type import DataEngineType
from .fourier import sin_cos_transform
from learning_machine.zoo import DATA_ENGINE_ZOO


@DATA_ENGINE_ZOO.regist()
class StringToDatetime(DataEngine):
    engine_type = [DataEngineType.SIDE_EFFECT]

    def __init__(self, col: str, format: str | None = None):
        """string to pd.datetiem object. Replace original string column to datetime object column

        Args:
            col (str): column name
            format (str | None, optional): datetime format. Defaults to None.
        """
        self.col = col
        self.format = format

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.col] = pd.to_datetime(data[self.col], format=self.format)
        return data


@DATA_ENGINE_ZOO.regist()
class DatetimeDayOfYearSinCos(DataEngine):
    engine_type = [DataEngineType.RETURN_NEW_PD]

    def __init__(self, col: str, prefix="day_of_year"):
        """Periodically transforms the day of year. Return 2 column {prefix}_{col}_sin/cos.

        Args:
            col (str): column name
            prefix (str, optional): return column prefix. Defaults to "day_of_year".
        """
        self.col = col
        self.prefix = prefix

    def __call__(self, data: pd.DataFrame):
        day_of_year = data[self.col].dt.dayofyear.to_numpy().astype(np.float64)
        day_of_year /= data[self.col].dt.is_leap_year.map({True: 366, False: 365})
        sinx, cosx = sin_cos_transform(day_of_year, 1)
        return pd.DataFrame(
            {
                f"{self.prefix}_{self.col}_sin": sinx,
                f"{self.prefix}_{self.col}_cos": cosx,
            },
            index=data.index,
        )


@DATA_ENGINE_ZOO.regist()
class DatetimeDayOfYear(DataEngine):
    engine_type = [DataEngineType.RETURN_NEW_PD]

    def __init__(self, col: str, prefix="day_of_year"):
        """get day of year from datetime column. Return {prefix}_{col}.

        Args:
            col (str): column name
            prefix (str, optional): return column prefix. Defaults to "day_of_year".
        """
        self.col = col
        self.prefix = prefix

    def __call__(self, data: pd.DataFrame):
        day_of_year = data[self.col].dt.dayofyear.to_numpy()
        return pd.DataFrame(
            {
                f"{self.prefix}_{self.col}": day_of_year,
            },
            index=data.index,
        )


@DATA_ENGINE_ZOO.regist()
class DatetimeMonthOfYear(DataEngine):
    engine_type = [DataEngineType.RETURN_NEW_PD]

    def __init__(self, col: str, prefix="month_of_year"):
        """get month of year from datetime column. Return {prefix}_{col}.

        Args:
            col (str): column name
            prefix (str, optional): return column prefix. Defaults to "month_of_year".
        """
        self.col = col
        self.prefix = prefix

    def __call__(self, data: pd.DataFrame):
        month_of_year = data[self.col].dt.month.to_numpy()
        return pd.DataFrame(
            {
                f"{self.prefix}_{self.col}": month_of_year,
            },
            index=data.index,
        )


@DATA_ENGINE_ZOO.regist()
class DatetimeMonthOfYearSinCos(DataEngine):
    engine_type = [DataEngineType.RETURN_NEW_PD]

    def __init__(self, col: str, prefix="month_of_year"):
        """Periodically transform the month of year. Return 2 column {prefix}_{col}_sin/cos

        Args:
            col (str): column name
            prefix (str, optional): return column prefix. Defaults to "month_of_year".
        """
        self.col = col
        self.prefix = prefix

    def __call__(self, data: pd.DataFrame):
        month_of_year = data[self.col].dt.month.to_numpy().astype(np.float64)
        sinx, cosx = sin_cos_transform(month_of_year, 12)
        return pd.DataFrame(
            {
                f"{self.prefix}_{self.col}_sin": sinx,
                f"{self.prefix}_{self.col}_cos": cosx,
            },
            index=data.index,
        )


@DATA_ENGINE_ZOO.regist()
class DatetimeDayOfMonth(DataEngine):
    engine_type = [DataEngineType.RETURN_NEW_PD]

    def __init__(self, col: str, prefix="day_of_month"):
        """get day of month from datetime column. Return {prefix}_{col}.

        Args:
            col (str): column name
            prefix (str, optional): return column prefix. Defaults to "day_of_month".
        """
        self.col = col
        self.prefix = prefix

    def __call__(self, data: pd.DataFrame):
        day_of_month = data[self.col].dt.day.to_numpy()
        return pd.DataFrame(
            {
                f"{self.prefix}_{self.col}": day_of_month,
            },
            index=data.index,
        )


@DATA_ENGINE_ZOO.regist()
class DatetimeDayOfMonthSinCos(DataEngine):
    engine_type = [DataEngineType.RETURN_NEW_PD]

    def __init__(self, col: str, prefix="day_of_month", norm=True):
        """Periodically transform the day of month. Return 2 column {prefix}_{col}_sin/cos.

        Args:
            col (str): column name
            prefix (str, optional): prefix of column. Defaults to "day_of_month".
            norm (bool, optional): preiod length. If False, prefiod fixed with 31, else, preiod length is end of month. Defaults to True.
        """
        self.col = col
        self.prefix = prefix
        self.norm = norm

    def __call__(self, data: pd.DataFrame):
        day_of_month = data[self.col].dt.day.to_numpy().astype(np.float64)
        if self.norm:
            day_of_month /= data[self.col].dt.days_in_month
        else:
            day_of_month /= 31

        sinx, cosx = sin_cos_transform(day_of_month, 1)
        return pd.DataFrame(
            {
                f"{self.prefix}_{self.col}_sin": sinx,
                f"{self.prefix}_{self.col}_cos": cosx,
            },
            index=data.index,
        )


@DATA_ENGINE_ZOO.regist()
class DatetimeDayOfWeek(DataEngine):
    engine_type = [DataEngineType.RETURN_NEW_PD]

    def __init__(self, col: str, prefix="day_of_week"):
        """get day of week from datetime column. Return {prefix}_{col}.

        Args:
            col (str): column name
            prefix (str, optional): return column prefix. Defaults to "day_of_week".
        """
        self.col = col
        self.prefix = prefix

    def __call__(self, data: pd.DataFrame):
        day_of_week = data[self.col].dt.day_of_week.to_numpy()
        return pd.DataFrame(
            {
                f"{self.prefix}_{self.col}": day_of_week,
            },
            index=data.index,
        )


@DATA_ENGINE_ZOO.regist()
class DatetimeDayOfWeekSinCos(DataEngine):
    engine_type = [DataEngineType.RETURN_NEW_PD]

    def __init__(self, col: str, prefix="day_of_week"):
        """Periodically transforms the day of week. Return 2 column {prefix}_{col}_sin/cos

        Args:
            col (str): column name
            prefix (str, optional): return column prefix. Defaults to "day_of_week".
        """
        self.col = col
        self.prefix = prefix

    def __call__(self, data: pd.DataFrame):
        day_of_week = data[self.col].dt.day.to_numpy().astype(np.float64)
        sinx, cosx = sin_cos_transform(day_of_week, 7)
        return pd.DataFrame(
            {
                f"{self.prefix}_{self.col}_sin": sinx,
                f"{self.prefix}_{self.col}_cos": cosx,
            },
            index=data.index,
        )


@DATA_ENGINE_ZOO.regist()
class DatetimeIsWeekend(DataEngine):
    engine_type = [DataEngineType.RETURN_NEW_PD]

    def __init__(self, col: str, prefix="is_weekend", include_sat=True):
        """get datetime is weekend. Return {prefix}_{col}

        Args:
            col (str): column name
            prefix (str, optional): return column prefix. Defaults to "is_weekend".
            include_sat (bool, optional): include saturday. If True, saturday is also weekend. Defaults to True.
        """
        self.col = col
        self.prefix = prefix
        self.include_sat = include_sat

    def __call__(self, data: pd.DataFrame):
        threshold = 5 if self.include_sat else 6
        is_weekend = data[self.col].dt.day_of_week.to_numpy() > threshold
        return pd.DataFrame(
            {
                f"{self.prefix}_{self.col}": is_weekend,
            },
            index=data.index,
        )
