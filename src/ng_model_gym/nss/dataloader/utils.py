# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from enum import Enum
from pathlib import Path
from typing import Sequence, Union

import torch

logger = logging.getLogger(__name__)

HDR_MAX = 65504.0


class DataLoaderMode(str, Enum):
    """Data loader mode, either train, validation or test."""

    TRAIN = "train"
    VAL = "validation"
    TEST = "test"


class DatasetType(str, Enum):
    """Dataset type, either .safetensors, .h5, .pt or .tftecord."""

    SAFETENSOR = ".safetensors"
    H5 = ".h5"
    PT = ".pt"
    TFRECORD = ".tfrecord"


class ToneMapperMode(str, Enum):
    """Supported modes for tone mapping."""

    REINHARD = "reinhard"
    KARIS = "karis"
    LOG = "log"
    LOG10 = "log10"
    LOG_NORM = "log_norm"
    ACES = "aces"


def tonemap_forward(
    x: torch.Tensor, max_val: float = 1.0, mode: ToneMapperMode = ToneMapperMode.KARIS
) -> torch.Tensor:
    """
    Reinhard:
    http://behindthepixels.io/assets/files/TemporalAA.pdf#page5

    Karis:
    http://graphicrants.blogspot.com/2013/12/tone-mapping.html
    """
    x = torch.max(x, torch.zeros_like(x))

    # Tonemap
    if mode == ToneMapperMode.REINHARD:
        x = x * (max_val / (1.0 + x))
    elif mode == ToneMapperMode.KARIS:
        # calculate dim based on x number of dimensions.
        luma = torch.max(x, dim=x.ndim - 3, keepdim=True).values
        x = x * (max_val / (1.0 + luma))
    elif mode == ToneMapperMode.LOG10:
        x = torch.log10(x + 1.0)
    elif mode == ToneMapperMode.LOG:
        x = torch.log(x + 1.0)
    elif mode == ToneMapperMode.LOG_NORM:
        log_norm_scale = 1.0 / torch.log(torch.tensor(HDR_MAX + 1.0))
        x = torch.log(x + 1.0) * log_norm_scale
    elif mode == ToneMapperMode.ACES:
        # See ACES in action and its inverse at https://www.desmos.com/calculator/n1lkpc6hwq
        k_aces_a = torch.tensor(2.51, dtype=x.dtype)
        k_aces_b = torch.tensor(0.03, dtype=x.dtype)
        k_aces_c = torch.tensor(2.43, dtype=x.dtype)
        k_aces_d = torch.tensor(0.59, dtype=x.dtype)
        k_aces_e = torch.tensor(0.14, dtype=x.dtype)
        x = (x * (k_aces_a * x + k_aces_b)) / (x * (k_aces_c * x + k_aces_d) + k_aces_e)
    else:
        raise ValueError(f"Tonemap: {mode} unsupported")

    return torch.clamp(x, 0.0, max_val)


def tonemap_inverse(
    x: torch.Tensor, max_val: float = 1.0, mode: ToneMapperMode = ToneMapperMode.KARIS
) -> torch.Tensor:
    """
    Reinhard:
    http://behindthepixels.io/assets/files/TemporalAA.pdf#page5

    Karis:
    http://graphicrants.blogspot.com/2013/12/tone-mapping.html
    """
    x = torch.max(x, torch.zeros_like(x))

    # Tonemap
    if mode == ToneMapperMode.REINHARD:
        x = torch.clamp(
            x,
            0.0,
            tonemap_forward(
                torch.tensor(HDR_MAX, dtype=x.dtype), mode=ToneMapperMode.REINHARD
            ),
        )
        x = x * (max_val / (1.0 - x))
    elif mode == ToneMapperMode.KARIS:
        x = torch.clamp(
            x,
            torch.tensor(0.0),
            tonemap_forward(
                torch.reshape(torch.tensor(HDR_MAX, dtype=x.dtype), (1, 1, 1, 1)),
                mode=ToneMapperMode.KARIS,
            ),
        )
        # calculate dim based on x number of dimensions.
        luma = torch.max(x, dim=x.ndim - 3, keepdim=True).values
        x = x * (max_val / (1.0 - luma))
    elif mode == ToneMapperMode.LOG:
        x = torch.exp(x) - 1.0
    elif mode == ToneMapperMode.LOG10:
        x = torch.pow(10, x) - 1.0
    elif mode == ToneMapperMode.LOG_NORM:
        log_norm_scale = 1.0 / torch.log(torch.tensor(HDR_MAX + 1.0))
        x = torch.exp(x / log_norm_scale) - 1.0
    elif mode == ToneMapperMode.ACES:
        # https://www.desmos.com/calculator/n1lkpc6hwq
        k_aces_a = torch.tensor(2.51, dtype=x.dtype)
        k_aces_b = torch.tensor(0.03, dtype=x.dtype)
        k_aces_c = torch.tensor(2.43, dtype=x.dtype)
        k_aces_d = torch.tensor(0.59, dtype=x.dtype)
        k_aces_e = torch.tensor(0.14, dtype=x.dtype)
        res = k_aces_d * x - k_aces_b
        res += torch.sqrt(
            x * x * (k_aces_d * k_aces_d - 4.0 * k_aces_e * k_aces_c)
            + x * (4.0 * k_aces_e * k_aces_a - 2.0 * k_aces_b * k_aces_d)
            + k_aces_b * k_aces_b
        )
        res /= 2.0 * k_aces_a - 2.0 * k_aces_c * x
        x = res
    else:
        raise ValueError(f"Tonemap: {mode} unsupported")

    return x


def _exclude_frames(frame_list_: Sequence, exclude_: Sequence):
    """Excludes from the supplied frame list anything that is contained in the exclude list.

    Args:
        frame_list_: Initial list of frames.
        exclude_: List of frames to exclude.

    Returns:
        frame_list but with any paths that are in the exclude list removed.
    """
    return [
        f for f in frame_list_ if not any(substring in f[0] for substring in exclude_)
    ]


def _get_frame_list_from_path(
    path: Path, extension: str = DatasetType.PT, n_frames: int = 1, keyword: str = ""
):
    """Returns a list of lists of sequential frames from the given path.

    Args:
        path: Path to a folder of saved data frames.
        extension: Expected extension of the saved data frames e.g. ".pt" or ".h5".
        n_frames: Number of sequential frames that will be used to make one sequence.
        keyword: Optional keyword that must be contained in the sub folder paths.

    Returns:
        List of Lists containing frame paths grouped into n_frames length sequences.
    """
    frames = []

    try:
        path_obj = Path(path)

        if (
            path_obj.is_file()
            and path_obj.exists()
            and str(path_obj).endswith(extension)
        ):
            frames.append(str(path))
            return frames

        if not path_obj.is_dir():
            # If we don't manually check this, the following globbing code will silently fail,
            # making it hard to track down the problem.
            raise ValueError(f"Path {path} is not an existing directory.")
    except Exception as e:
        # Propagate any IO errors (e.g. lack of permissions)
        raise ValueError(f"Error while accessing path {path}.") from e

    seq_dirs = sorted(
        list(
            {
                p.parent
                for p in sorted(list(path_obj.rglob(f"*{extension}")))
                if p.is_file()
            }
        )
    )
    for root in seq_dirs:
        if keyword not in str(root):
            continue

        images = sorted(
            [str(p) for p in root.iterdir() if p.is_file() and p.suffix == extension]
        )

        # Take the whole sequence if n_frames is -1
        if n_frames == -1:
            frames.append(images)
        else:
            for idx in range(0, len(images) - (n_frames - 1)):
                seq_in = []
                for i in range(n_frames):
                    seq_in.append(images[idx + i])
                frames.append(seq_in)

    return frames


def get_frame_list(
    path: Union[Sequence[Path], Path],
    n_frames: int = 1,
    exclude: Sequence = None,
    extension: DatasetType = DatasetType.PT,
):
    """Gets a List of Lists of sequential frame paths from the given folder path.

    Args:
        path: Path to a single folder of files or a List of folder Paths.
        n_frames: Number of frames that will be used to make one sequence.
        exclude: List of files to exclude.
        extension: The extension type of the saved data e.g. ".pt" or ".h5".

    Returns:
        List of Lists containing frame paths grouped into n_frames length sequences.
    """
    if exclude is None:
        exclude = []

    if isinstance(path, list):
        frame_list = []
        for p in path:
            frame_list.extend(
                _get_frame_list_from_path(
                    p, extension=extension.value, n_frames=n_frames, keyword=""
                )
            )
    else:
        frame_list = _get_frame_list_from_path(
            path, extension=extension.value, n_frames=n_frames, keyword=""
        )
    return _exclude_frames(frame_list, exclude)
