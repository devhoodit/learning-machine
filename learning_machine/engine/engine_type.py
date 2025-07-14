from enum import Enum, auto


class DataEngineType(Enum):
    RETURN_NEW_PD = auto()
    SIDE_EFFECT = auto()

    NDArr = auto()
    Dataframe = auto()
