# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
import random
from typing import Type

import numpy as np
import torch
from torch.utils.data import Dataset

from ng_model_gym.core.data.dataset_registry import DATASET_REGISTRY, get_dataset_key
from ng_model_gym.core.data.utils import DataLoaderMode, DatasetType
from ng_model_gym.core.utils.config_model import ConfigModel

logger = logging.getLogger(__name__)


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
        loader_mode: What mode the dataloader should be set to.
        trace_mode: Whether the dataloader returned is for tracing at export.
            Should be False for QAT.

    Returns:
        PyTorch DataLoader object ready to produce data as configured by passed in parameters.
    """

    dataset = get_dataset(config_params, loader_mode)

    if trace_mode:
        logger.debug(
            "Loading data in trace mode with batch_size of 2 (for FP32/PTQ export)"
        )
        batch_size = 2  # Avoid 0/1 Specialization Problem in PT2 Export
    else:
        batch_size = (
            1 if loader_mode == DataLoaderMode.TEST else config_params.train.batch_size
        )

    # Shuffle only when training.
    shuffle = loader_mode == DataLoaderMode.TRAIN

    if (
        config_params.dataset.recurrent_samples is not None
        and config_params.dataset.recurrent_samples < 2
        and loader_mode == DataLoaderMode.TRAIN
    ):
        raise ValueError(
            "If set, number of recurrent samples must be greater than 1 for training."
        )

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
