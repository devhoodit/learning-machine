# ------------------------------------------------------------------------
# Modified from detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from pathlib import Path
import importlib.abc
import importlib.util

__CLONE_PROJECT_ROOT = Path(__file__).resolve().parent.parent / "projects"


def inject_projects(dir: Path):
    if not dir.is_dir():
        ValueError("inject projects dir need projects directory path")

    class ProjectFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if not fullname.startswith("learning_machine.projects."):
                return
            project_name = fullname.split(".")[-1]
            project_dir = dir.joinpath(project_name)
            if not project_dir:
                return
            target_file = project_dir.joinpath("__init__.py")
            if not target_file.is_file():
                return
            return importlib.util.spec_from_file_location(fullname, target_file)

    import sys

    sys.meta_path.append(ProjectFinder())


def preload_projects(dir: Path | str) -> list[str]:
    dir = Path(dir)
    if not dir.is_dir():
        ValueError("preload projects dir need projects directory path")

    preloaded_projects_name = []
    for project_dir in dir.iterdir():
        if not project_dir.is_dir():
            continue
        project_name = project_dir.name
        target_file = project_dir.joinpath("__init__.py")
        if not target_file.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            f"learning_machine.projects.{project_name}", target_file
        )
        if spec is None:
            continue
        if spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        preloaded_projects_name.append(project_name)
    return preloaded_projects_name


def prelaod_builtin_projects() -> list[str]:
    projects = []
    if __CLONE_PROJECT_ROOT.is_dir():
        projects += preload_projects(__CLONE_PROJECT_ROOT)
    return projects


if __CLONE_PROJECT_ROOT.is_dir():
    inject_projects(__CLONE_PROJECT_ROOT)
