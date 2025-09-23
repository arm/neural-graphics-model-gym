# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Type

from torch.utils.data import Dataset

from ng_model_gym.core.utils.registry import Registry


def _validate_dataset(dataset_class: Type[Dataset]) -> None:
    """Ensure only valid datasets are added to the registry."""
    # Check dataset is a class.
    if not isinstance(dataset_class, type):
        raise TypeError(f"Registered object {dataset_class} must be a class.")

    # Check dataset inherits from torch.utils.data.Dataset.
    if not issubclass(dataset_class, Dataset):
        raise TypeError(
            f"{dataset_class.__name__} must inherit from torch.utils.data.Dataset."
        )

    # Make sure __len__() has been implemented.
    if "__len__" not in dataset_class.__dict__:
        raise TypeError(f"{dataset_class.__name__} must override __len__(self).")

    # Make sure __getitem__() has been implemented.
    if "__getitem__" not in dataset_class.__dict__:
        raise TypeError(
            f"{dataset_class.__name__} must override __getitem__(self, index)."
        )


DATASET_REGISTRY: Registry[Dataset] = Registry(
    registry_name="dataset", validator=_validate_dataset
)


def register_dataset(name: str, version: Optional[str] = None):
    """
    Helper function to add a new dataset to the dataset registry,
    using a new unique identifier as the key, with optional version.
    Example:
        >>> @register_dataset(name="NSS", version="1")
        >>> class NSS_Dataset(Dataset):
        >>>     pass
    """
    key = f"{name}-v{version}" if version is not None else name

    return DATASET_REGISTRY.register(key.lower())
