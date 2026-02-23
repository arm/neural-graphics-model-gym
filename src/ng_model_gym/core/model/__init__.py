# SPDX-FileCopyrightText: <text>Copyright 2025-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from .base_ng_model import BaseNGModel
from .dense_warp_utils import (
    backward_warp_nearest,
    bilinear_oob_zero,
    catmull_rom_warp,
    dense_image_warp,
    interpolate_bilinear,
    interpolate_bilinear_w_zero_pad,
)
from .graphics_utils import (
    compute_jitter_tile_offset,
    compute_luminance,
    fixed_normalize_mvs,
    generate_lr_to_hr_lut,
    length,
    lerp,
    normalize_mvs,
    normalize_mvs_fixed,
    swizzle,
)
from .layers.conv_block import ConvBlock
from .layers.dense_warp import DenseWarp
from .layers.resampling import DownSampling2D, UpSampling2D, ZeroUpsample
from .model_factory import create_model, get_model_from_config
from .model_registry import (
    _validate_model,
    get_model_key,
    MODEL_REGISTRY,
    register_model,
)
from .model_tracer import model_tracer
from .shaders.slang_utils import load_slang_module
