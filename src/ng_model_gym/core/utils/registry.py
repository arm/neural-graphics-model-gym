# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Generic, List, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """
    Registry to register an object.
    Example:
        >>> MODEL_REGISTRY: Registry[BaseNGModel] = Registry("Model")
        >>> @MODEL_REGISTRY.register("NSS_v1")
        >>> class NSS_V1(BaseNGModel):
        >>>         pass
        >>> nss_model = MODEL_REGISTRY.get("NSS_v1")
        >>> print(f"All available models: {MODEL_REGISTRY.list_registered()}")
    """

    def __init__(self, registry_name: str):
        self._name = registry_name
        self._registry: Dict[str, Type[T]] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Register a new object."""

        def decorator(class_obj: Type[T]) -> Type[T]:
            if name in self._registry:
                raise KeyError(f"{self._name} {name} is already registered.")
            self._registry[name] = class_obj
            return class_obj

        return decorator

    def get(self, name: str) -> Type[T]:
        """Return a registered object, if the name exists as a key."""
        try:
            return self._registry[name]
        except KeyError as e:
            raise KeyError(f"{self._name} {name} is not registered.") from e

    def list_registered(self) -> List[str]:
        """Return all keys in the registry."""
        return sorted(self._registry.keys())
