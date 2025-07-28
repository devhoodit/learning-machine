from __future__ import annotations
from typing import Generic, TypeVar
from abc import ABC, abstractmethod

from learning_machine.zoo import DATA_ENGINE_ZOO


def create_engines_from_config(config: list[dict]) -> list[DataEngine]:
    """Create engines from engine config list.

    Args:
        config (list[dict]): engines config

    Returns:
        list[DataEngine]: engines
    """
    engines = []
    for e in config:
        engine_name = next(iter(e))
        args = e[engine_name]
        engine = DATA_ENGINE_ZOO.get(engine_name)
        engine_ins = engine.from_config(args)
        engines.append(engine_ins)
    return engines


T = TypeVar("T")
U = TypeVar("U")


class DataEngine(ABC, Generic[T, U]):
    """Data engine interface."""

    engine_type = []

    @abstractmethod
    def __call__(self, data: T) -> U:
        """Process data

        Args:
            data (T)

        Returns:
            U: processed data
        """
        pass

    @classmethod
    def from_config(cls, config: dict) -> DataEngine:
        """Create engine from config.

        Args:
            config (dict): engine configuration information

        Returns:
            DataEngine: data engine
        """
        return cls(**config)


@DATA_ENGINE_ZOO.regist()
class SequentialEngine(DataEngine, Generic[T, U]):
    """Apply engines sequentially.
    data -> engine1 -> data1 -> engine2 -> data2
    """

    def __init__(self, engines: list[DataEngine]):
        """
        Args:
            engines (list[DataEngine]): engines
        """
        super().__init__()
        self.engines = engines

    def __call__(self, data: T) -> U:
        for engine in self.engines:
            data = engine(data)
        return data  # type: ignore
