# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
from typing import Dict, Tuple

import torch
import torchvision.transforms.functional as transforms

from ng_model_gym.usecases.nfru.data.naming import (
    convert_str_offset_to_int,
    DataVariable,
)

_SCALAR_KEYS = {
    "img_id",
    "ViewProj",
    "NearPlane",
    "FarPlane",
    "FovX",
    "FovY",
    "infinite_zFar",
    "outDims",
    "inDims",
    "seq_id",
    "seq",
    "InverseY",
    "exposure",
}

_MOTION_PREFIXES = ("mv_", "flow_", "sy_", "cm_")
NFRU_DEFAULT_SHAPE_AUG_NUM_SHAPES = (30, 10)
NFRU_DEFAULT_SHAPE_AUG_MAX_SIZE = (15, 50)
NFRU_DEFAULT_SHAPE_AUG_MAX_DISPLACEMENT = (15, 20)
NFRU_DEFAULT_SHAPE_AUG_PROBABILITY = 0.15
NFRU_DEFAULT_BRIGHTNESS_SHAPE_AUG_PROBABILITY = 0.15
_NFRU_DEFAULT_BRIGHTNESS_SHAPE_AUG_NUM_SHAPES = 7
_NFRU_DEFAULT_BRIGHTNESS_SHAPE_AUG_MAX_SIZE = 200
_NFRU_DEFAULT_BRIGHTNESS_SHAPE_AUG_MAX_DISPLACEMENT = 70

_NFRU_SHAPE_AUG_MIN_SHAPE_SIZE = 10

_NFRU_BRIGHTNESS_SHAPE_AUG_MIN_SHAPE_SIZE = 30

_NFRU_BRIGHTNESS_SCALE_BRIGHTEN_BASE = 1.75
_NFRU_BRIGHTNESS_SCALE_DARKEN_BASE = 0.25
_NFRU_BRIGHTNESS_SCALE_VARIATION = 0.25

_NFRU_TEMPORAL_INTERPOLATION_DIVISOR = 2
_NFRU_AUGMENTATION_MARGIN_PADDING = 1
_NFRU_RANDOM_BINARY_HIGH = 2
_NFRU_FLIP_MATRIX_SIZE = 4


def _split_scalar_entries(
    data_frame_in: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    scalars: Dict[str, torch.Tensor] = {}
    tensors: Dict[str, torch.Tensor] = {}
    for key, value in data_frame_in.items():
        non_concrete = DataVariable(key).generate_non_concrete_variable()
        if non_concrete in _SCALAR_KEYS:
            scalars[key] = value
        else:
            tensors[key] = value
    return scalars, tensors


def _stamp_shape(
    canvas: torch.Tensor,
    y: int,
    x: int,
    size_h: int,
    size_w: int,
    color: torch.Tensor,
    is_rectangle: bool,
) -> None:
    """Stamp a rectangle when `is_rectangle` is True, otherwise stamp an ellipse."""
    region = canvas[:, y : y + size_h, x : x + size_w]
    if is_rectangle:
        canvas[:, y : y + size_h, x : x + size_w] = color[:, None, None]
        return

    cy, cx = size_h / 2.0, size_w / 2.0
    ys = torch.arange(size_h, device=canvas.device, dtype=torch.float32)
    xs = torch.arange(size_w, device=canvas.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    ellipse_mask = (
        (((yy - cy) / max(cy, 1.0)) ** 2 + ((xx - cx) / max(cx, 1.0)) ** 2) <= 1.0
    ).unsqueeze(0)
    canvas[:, y : y + size_h, x : x + size_w] = torch.where(
        ellipse_mask, color[:, None, None], region
    )


def apply_shape_augmentation(
    rgb_m1: torch.Tensor,
    rgb_t: torch.Tensor,
    rgb_p1: torch.Tensor,
    num_shapes: int,
    max_shape_size: int,
    max_displacement: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stamp temporally consistent coloured shapes into the RGB triplet."""
    channels, height, width = rgb_m1.shape
    rgb_m1 = rgb_m1.clone()
    rgb_t = rgb_t.clone()
    rgb_p1 = rgb_p1.clone()

    margin = max_displacement + max_shape_size + _NFRU_AUGMENTATION_MARGIN_PADDING
    if height <= 2 * margin or width <= 2 * margin:
        return rgb_m1, rgb_t, rgb_p1

    for _ in range(num_shapes):
        size_h = random.randint(_NFRU_SHAPE_AUG_MIN_SHAPE_SIZE, max_shape_size)
        size_w = random.randint(_NFRU_SHAPE_AUG_MIN_SHAPE_SIZE, max_shape_size)
        color = torch.rand(channels, dtype=rgb_m1.dtype, device=rgb_m1.device)

        y_m1 = random.randint(margin, height - size_h - margin)
        x_m1 = random.randint(margin, width - size_w - margin)
        dy = random.randint(-max_displacement, max_displacement)
        dx = random.randint(-max_displacement, max_displacement)
        y_p1 = y_m1 + dy
        x_p1 = x_m1 + dx
        y_t = y_m1 + dy // _NFRU_TEMPORAL_INTERPOLATION_DIVISOR
        x_t = x_m1 + dx // _NFRU_TEMPORAL_INTERPOLATION_DIVISOR
        is_rectangle = bool(random.getrandbits(1))

        _stamp_shape(rgb_m1, y_m1, x_m1, size_h, size_w, color, is_rectangle)
        _stamp_shape(rgb_t, y_t, x_t, size_h, size_w, color, is_rectangle)
        _stamp_shape(rgb_p1, y_p1, x_p1, size_h, size_w, color, is_rectangle)

    return rgb_m1, rgb_t, rgb_p1


def _stamp_shape_brightness(
    canvas: torch.Tensor,
    y: int,
    x: int,
    size_h: int,
    size_w: int,
    brightness_scale: float,
    is_rectangle: bool,
) -> None:
    """Apply brightness to a rectangle when `is_rectangle` is True, else to an ellipse."""
    region = canvas[:, y : y + size_h, x : x + size_w]
    adjusted = region * brightness_scale
    if is_rectangle:
        canvas[:, y : y + size_h, x : x + size_w] = adjusted
        return

    cy, cx = size_h / 2.0, size_w / 2.0
    ys = torch.arange(size_h, device=canvas.device, dtype=torch.float32)
    xs = torch.arange(size_w, device=canvas.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    ellipse_mask = (
        (((yy - cy) / max(cy, 1.0)) ** 2 + ((xx - cx) / max(cx, 1.0)) ** 2) <= 1.0
    ).unsqueeze(0)
    canvas[:, y : y + size_h, x : x + size_w] = torch.where(
        ellipse_mask, adjusted, region
    )


def apply_shape_brightness_augmentation(
    rgb_m1: torch.Tensor,
    rgb_t: torch.Tensor,
    rgb_p1: torch.Tensor,
    num_shapes: int = _NFRU_DEFAULT_BRIGHTNESS_SHAPE_AUG_NUM_SHAPES,
    max_shape_size: int = _NFRU_DEFAULT_BRIGHTNESS_SHAPE_AUG_MAX_SIZE,
    max_displacement: int = _NFRU_DEFAULT_BRIGHTNESS_SHAPE_AUG_MAX_DISPLACEMENT,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply temporally consistent local brightness changes to the RGB triplet."""
    _, height, width = rgb_m1.shape
    rgb_m1 = rgb_m1.clone()
    rgb_t = rgb_t.clone()
    rgb_p1 = rgb_p1.clone()

    margin = max_displacement + max_shape_size + _NFRU_AUGMENTATION_MARGIN_PADDING
    if height <= 2 * margin or width <= 2 * margin:
        return rgb_m1, rgb_t, rgb_p1

    for _ in range(num_shapes):
        size_h = random.randint(
            _NFRU_BRIGHTNESS_SHAPE_AUG_MIN_SHAPE_SIZE, max_shape_size
        )
        size_w = random.randint(
            _NFRU_BRIGHTNESS_SHAPE_AUG_MIN_SHAPE_SIZE, max_shape_size
        )
        y_m1 = random.randint(margin, height - size_h - margin)
        x_m1 = random.randint(margin, width - size_w - margin)
        dy = random.randint(-max_displacement, max_displacement)
        dx = random.randint(-max_displacement, max_displacement)
        y_p1 = y_m1 + dy
        x_p1 = x_m1 + dx
        y_t = y_m1 + dy // _NFRU_TEMPORAL_INTERPOLATION_DIVISOR
        x_t = x_m1 + dx // _NFRU_TEMPORAL_INTERPOLATION_DIVISOR

        # False means ellipse
        is_rectangle = False

        if bool(random.getrandbits(1)):
            brightness_scale = _NFRU_BRIGHTNESS_SCALE_BRIGHTEN_BASE + random.uniform(
                0.0, _NFRU_BRIGHTNESS_SCALE_VARIATION
            )
        else:
            brightness_scale = _NFRU_BRIGHTNESS_SCALE_DARKEN_BASE - random.uniform(
                0.0, _NFRU_BRIGHTNESS_SCALE_VARIATION
            )

        _stamp_shape_brightness(
            rgb_m1, y_m1, x_m1, size_h, size_w, brightness_scale, is_rectangle
        )
        _stamp_shape_brightness(
            rgb_t, y_t, x_t, size_h, size_w, brightness_scale, is_rectangle
        )
        _stamp_shape_brightness(
            rgb_p1, y_p1, x_p1, size_h, size_w, brightness_scale, is_rectangle
        )

    return rgb_m1, rgb_t, rgb_p1


def _augment_motion_tensor(
    tensor: torch.Tensor, flip_horz: bool, flip_vert: bool
) -> torch.Tensor:
    v, u = torch.split(tensor, split_size_or_sections=1, dim=1)
    if flip_horz:
        v = transforms.hflip(v)
        u = -transforms.hflip(u)
    if flip_vert:
        v = -transforms.vflip(v)
        u = transforms.vflip(u)
    return torch.concat([v, u], dim=1)


def _augment_tensor(
    key: str, tensor: torch.Tensor, flip_horz: bool, flip_vert: bool
) -> torch.Tensor:
    if not (flip_horz or flip_vert):
        return tensor
    if any(key.startswith(prefix) for prefix in _MOTION_PREFIXES):
        return _augment_motion_tensor(tensor, flip_horz, flip_vert)
    augmented = tensor
    if flip_horz:
        augmented = transforms.hflip(augmented)
    if flip_vert:
        augmented = transforms.vflip(augmented)
    return augmented


def _apply_shape_augmentations(
    data_frame: Dict[str, torch.Tensor],
    shape_aug: bool,
    shape_aug_num_shapes: list[int] | tuple[int, ...] | int,
    shape_aug_max_size: list[int] | tuple[int, ...] | int,
    shape_aug_max_displacement: list[int] | tuple[int, ...] | int,
    shape_aug_probability: float,
    brightness_shape_aug_probability: float,
) -> Dict[str, torch.Tensor]:
    if not shape_aug:
        return data_frame

    data_frame = data_frame.copy()
    rgb_m1_key = next(
        (
            key
            for key in data_frame
            if "rgb" in key and convert_str_offset_to_int(key.split("_")[-1]) < 0
        ),
        None,
    )
    rgb_t_key = next(
        (
            key
            for key in data_frame
            if "rgb" in key and convert_str_offset_to_int(key.split("_")[-1]) == 0
        ),
        None,
    )
    rgb_p1_key = next(
        (
            key
            for key in data_frame
            if "rgb" in key and convert_str_offset_to_int(key.split("_")[-1]) > 0
        ),
        None,
    )
    if rgb_m1_key is None or rgb_t_key is None or rgb_p1_key is None:
        return data_frame

    num_shapes_cfg = (
        [shape_aug_num_shapes]
        if isinstance(shape_aug_num_shapes, int)
        else list(shape_aug_num_shapes)
    )
    max_size_cfg = (
        [shape_aug_max_size]
        if isinstance(shape_aug_max_size, int)
        else list(shape_aug_max_size)
    )
    max_displacement_cfg = (
        [shape_aug_max_displacement]
        if isinstance(shape_aug_max_displacement, int)
        else list(shape_aug_max_displacement)
    )

    if torch.rand(()).item() < shape_aug_probability:
        for num_shapes, max_size, max_displacement in zip(
            num_shapes_cfg, max_size_cfg, max_displacement_cfg, strict=False
        ):
            aug_m1, aug_t, aug_p1 = apply_shape_augmentation(
                data_frame[rgb_m1_key].squeeze(0),
                data_frame[rgb_t_key].squeeze(0),
                data_frame[rgb_p1_key].squeeze(0),
                num_shapes=num_shapes,
                max_shape_size=max_size,
                max_displacement=max_displacement,
            )
            data_frame[rgb_m1_key] = aug_m1.unsqueeze(0)
            data_frame[rgb_t_key] = aug_t.unsqueeze(0)
            data_frame[rgb_p1_key] = aug_p1.unsqueeze(0)

    if torch.rand(()).item() < brightness_shape_aug_probability:
        aug_m1, aug_t, aug_p1 = apply_shape_brightness_augmentation(
            data_frame[rgb_m1_key].squeeze(0),
            data_frame[rgb_t_key].squeeze(0),
            data_frame[rgb_p1_key].squeeze(0),
        )
        data_frame[rgb_m1_key] = aug_m1.unsqueeze(0)
        data_frame[rgb_t_key] = aug_t.unsqueeze(0)
        data_frame[rgb_p1_key] = aug_p1.unsqueeze(0)

    return data_frame


def process_data(
    data_frame_in: Dict[str, torch.Tensor],
    augment: bool,
    shape_aug: bool,
    shape_aug_num_shapes: list[int] | tuple[int, ...] | int,
    shape_aug_max_size: list[int] | tuple[int, ...] | int,
    shape_aug_max_displacement: list[int] | tuple[int, ...] | int,
    shape_aug_probability: float,
    brightness_shape_aug_probability: float,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Prepare NFRU model inputs and the ground-truth target frame."""
    data_frame = _apply_shape_augmentations(
        data_frame_in,
        shape_aug=augment and shape_aug,
        shape_aug_num_shapes=shape_aug_num_shapes,
        shape_aug_max_size=shape_aug_max_size,
        shape_aug_max_displacement=shape_aug_max_displacement,
        shape_aug_probability=shape_aug_probability,
        brightness_shape_aug_probability=brightness_shape_aug_probability,
    )

    scalars, tensor_data = _split_scalar_entries(data_frame)

    flip_horz = flip_vert = False
    if augment:
        flip_horz = bool(
            torch.randint(
                low=0,
                high=_NFRU_RANDOM_BINARY_HIGH,
                size=(),
                dtype=torch.int32,
                device="cpu",
            ).item()
        )
        flip_vert = bool(
            torch.randint(
                low=0,
                high=_NFRU_RANDOM_BINARY_HIGH,
                size=(),
                dtype=torch.int32,
                device="cpu",
            ).item()
        )

    if augment:
        flip_matrix = torch.eye(
            _NFRU_FLIP_MATRIX_SIZE,
            device=scalars["ViewProj_m1"].device,
            dtype=scalars["ViewProj_m1"].dtype,
        ).unsqueeze(0)
        flip_matrix[0, 0, 0] = -1 if flip_horz else 1
        flip_matrix[0, 1, 1] = -1 if flip_vert else 1
        for key in list(scalars.keys()):
            if "ViewProj" in key:
                scalars[key] = flip_matrix @ scalars[key]

    scalars["MotionMat"] = torch.stack(
        [
            scalars["ViewProj_m1"].squeeze(0)
            @ torch.linalg.inv(scalars["ViewProj_p1"].squeeze(0)),
            scalars["ViewProj_p1"].squeeze(0)
            @ torch.linalg.inv(scalars["ViewProj_m1"].squeeze(0)),
        ]
    )

    net_inputs: Dict[str, torch.Tensor] = dict(scalars)
    y_true: torch.Tensor | None = None

    for key, tensor in tensor_data.items():
        processed = _augment_tensor(key, tensor, flip_horz, flip_vert)

        if "rgb" in key:
            index = convert_str_offset_to_int(key.split("_")[-1])
            is_t_index = index == 0
            if not is_t_index:
                net_inputs[key] = torch.squeeze(processed, dim=0)
            elif "y_true" not in net_inputs:
                y_true = torch.squeeze(processed, dim=0)
                net_inputs["y_true"] = y_true
        else:
            net_inputs[key] = torch.squeeze(processed, dim=0)

    if y_true is None:
        raise ValueError("Missing rgb target frame in safetensors slice.")

    net_inputs = {key: value.to(torch.float32) for key, value in net_inputs.items()}
    return net_inputs, y_true.to(torch.float32)
