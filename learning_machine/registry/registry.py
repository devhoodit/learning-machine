from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self):
        self._registry: dict[str, Any] = {}

    def get(self, name: str) -> T:
        return self._registry[name]

    def regist(self, name: str | None = None):
        def wrapper(wrapped_cls):
            cls_name = name
            if cls_name is None:
                cls_name = wrapped_cls.__name__
            if self._registry.get(cls_name) is not None:
                raise ValueError(f"duplicated model name. {name} is already registed")
            self._registry[cls_name] = wrapped_cls
            return wrapped_cls

        return wrapper

    def get_registry(self):
        return self._registry
