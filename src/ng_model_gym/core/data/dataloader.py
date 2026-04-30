# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import random
from typing import Type

import numpy as np
import torch
from torch.utils.data import Dataset

from ng_model_gym.core.config.config_model import ConfigModel
from ng_model_gym.core.data.data_utils import DataLoaderMode, DatasetType
from ng_model_gym.core.data.dataset_registry import DATASET_REGISTRY, get_dataset_key

logger = logging.getLogger(__name__)

_TRACE_MODE_DESCRIPTION = "trace mode (for FP32/PTQ export)"


def seed_worker(worker_id):  # pylint: disable=unused-argument
    """
    Make random functions that are run from workers be deterministic -
    see https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    # Torch will automatically set a seed for this worker based on a random number generated
    # from the torch instance running in the main process, but it doesn't handle seeds for other
    # libraries that we use such as numpy.
    # So we copy the auto-generated torch seed to these other libraries.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset_from_config(params: ConfigModel) -> Type[Dataset]:
    """Return the registered dataset class from the dataset registry,
    using the name and version supplied in the config file."""

    dataset_name = params.dataset.name
    dataset_version = params.dataset.version

    dataset_key = get_dataset_key(dataset_name, dataset_version)

    return DATASET_REGISTRY.get(dataset_key)


def get_dataset(
    params: ConfigModel, loader_mode: DataLoaderMode = DataLoaderMode.TRAIN
):
    """Return the dataset to be used."""

    dataset_cls = get_dataset_from_config(params)
    if hasattr(params.dataset, "extension"):
        extension = params.dataset.extension
    else:
        raise ValueError("Dataset extension not specified in config parameters.")

    try:
        extension = DatasetType(extension)
    except ValueError as e:
        raise ValueError(
            f"Unsupported dataset extension: {extension}. "
            f"Supported extensions are: {[ext.value for ext in DatasetType]}"
        ) from e

    dataset = dataset_cls(params, loader_mode=loader_mode, extension=extension)

    return dataset


def _get_dataloader_batch_size(
    configured_batch_size: int,
    dataset_size: int,
    loader_mode: DataLoaderMode,
    trace_mode: bool,
):
    """
    Gets the batch size that should be used for DataLoader objects with
    specified attributes.

    Args:
        configured_batch_size: Batch size from user configuration (assumed
           positive). Throws ValueError if the configured batch size is out of
           range.
        dataset_size: Length of the dataset about to be loaded.
        loader_mode: Dataloader setting - TRAIN, VALIDATION or TEST.
        trace_mode: Whether the related dataloader returned is for tracing at
            export. Should be False for QAT.
    """
    if loader_mode == DataLoaderMode.TEST:
        if trace_mode:
            raise ValueError(
                "TEST dataloader mode cannot be used "
                f"together with {_TRACE_MODE_DESCRIPTION}."
            )

        batch_size = 1
    elif configured_batch_size <= dataset_size:
        batch_size = configured_batch_size
    else:
        raise ValueError(
            f"Batch size ({configured_batch_size}) is larger "
            "than the dataset. Reduce batch size to "
            f"{dataset_size} or less."
        )

    if trace_mode and batch_size == 1:
        # Avoid 0/1 Specialization Problem in PT2 Export
        raise ValueError(
            f"In {_TRACE_MODE_DESCRIPTION}, the batch size must be 2 or more."
        )

    if batch_size != configured_batch_size:
        logger.warning(
            f"Using batch size of {batch_size} "
            f"(configured batch size = {configured_batch_size})"
        )

    return batch_size


def get_dataloader(
    config_params: ConfigModel,
    num_workers=1,
    prefetch_factor=1,
    loader_mode: DataLoaderMode = DataLoaderMode.TRAIN,
    trace_mode: bool = False,  # Note: should be False for QAT
):
    """Return the desired DataLoader.

    Args:
        config_params: Configuration parameters.
        num_workers: Number of workers for the DataLoader to use.
        prefetch_factor: How many batches each worker should try to preload.
        loader_mode: Dataloader setting to apply - TRAIN, VALIDATION or TEST.
        trace_mode: Whether the dataloader returned is for tracing at export.
            Should be False for QAT.

    Returns:
        PyTorch DataLoader object ready to produce data as configured by passed in parameters.
    """
    dataset = get_dataset(config_params, loader_mode)
    dataset_size = len(dataset)

    if dataset_size <= 0:
        raise ValueError("Cannot process empty dataset.")

    batch_size = _get_dataloader_batch_size(
        config_params.train.batch_size, dataset_size, loader_mode, trace_mode
    )

    messages = ["Loading data"]

    if trace_mode:
        messages.append(f"in {_TRACE_MODE_DESCRIPTION}")

    messages.append(f"with a batch size of {batch_size}")

    logger.debug(" ".join(messages))

    # Shuffle only when training.
    shuffle = loader_mode == DataLoaderMode.TRAIN

    g = torch.Generator()
    g.manual_seed(config_params.train.seed)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=g,
    )

    if config_params.dataset.health_check:
        if hasattr(dataset, "health_check"):
            dataset.health_check(dataloader)
        else:
            logger.warning(
                f"Dataset health check is enabled but the health_check() method is not "
                f"implemented for {type(dataset).__name__}. Skipping health check."
            )

    return dataloader
