import yaml
from pathlib import Path
from typing import Any
from learning_machine.engine import (
    create_engines_from_config,
    DataEngine,
    SequentialEngine,
)
from learning_machine.projects import preload_projects
from dataclasses import dataclass


@dataclass
class Bundle:
    """
    Bundle of components

    Attributes:
        data_engines (list[DataEngine]): List of engines.
        data_engine (DataEngine | None): Apply Sequential engine with list of engines.
    """

    # data: pd.DataFrame
    data_engines: list[DataEngine]
    data_engine: DataEngine | None
    model: Any


def create_from_config(path_or_config: dict | str) -> Bundle:
    """Create bundle from config or config file.

    Args:
        path_or_config (dict | str): config dict or config file path

    Returns:
        Bundle: engine and model bundle
    """
    if not path_or_config:
        raise ValueError()

    if isinstance(path_or_config, str):
        with open(path_or_config, "r") as f:
            path_or_config = yaml.safe_load(f)

    config: dict = path_or_config  # type: ignore

    if config.get("projects"):
        for preload_path in config["projects"]:
            preload_projects(Path(preload_path))

    # data engine
    data_engine = None
    data_engines = []
    if config.get("data_engine"):
        data_engines = create_engines_from_config(config["data_engine"])
        data_engine = SequentialEngine(data_engines)

    return Bundle(
        data_engines,
        data_engine,
        None,
    )


def get_parameter(params: dict) -> dict[str, str]:
    save_dict = {}
    for k, v in params.items():
        if k != "self":
            continue
        save_dict[k] = str(v)
    return save_dict
