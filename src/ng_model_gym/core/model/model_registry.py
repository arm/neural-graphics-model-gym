# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import inspect
from typing import Optional, Type

from torch import nn

from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.utils.registry import Registry


def _validate_model(model_class: Type[BaseNGModel]) -> None:
    """Ensure only valid models are added to the registry."""
    # Check model is a class.
    if not isinstance(model_class, type):
        raise TypeError(f"Registered object {model_class} must be a class.")

    # Check model inherits from BaseNGModel.
    if not issubclass(model_class, BaseNGModel):
        raise TypeError(f"{model_class.__name__} must inherit from BaseNGModel.")

    # Check all abstract methods are implemented e.g. get_neural_network().
    if inspect.isabstract(model_class):
        raise TypeError(
            "Make sure all abstract methods, e.g. get_neural_network(self), are implemented."
        )

    model_forward = getattr(model_class, "forward", model_class.forward)
    base_forward = getattr(nn.Module, "forward")

    # Make sure forward() is defined.
    if not callable(model_forward):
        raise TypeError(
            f"{model_class.__name__} must define a forward(self, ...) method."
        )

    # Make sure forward() has been implemented.
    if model_forward is base_forward:
        raise TypeError(f"{model_class.__name__} must override forward().")


MODEL_REGISTRY: Registry[BaseNGModel] = Registry(
    registry_name="model", validator=_validate_model
)


def register_model(name: str, version: Optional[str] = None):
    """
    Helper function to add a new model to the model registry,
    using a new unique identifier as the key, with optional version.
    Example:
        >>> @register_model(name="NSS", version="1")
        >>> class NSS_Model(BaseNeuralGraphicsPipeline)
        >>>     pass
    """
    key = f"{name}-v{version}" if version is not None else name

    return MODEL_REGISTRY.register(key.lower())
