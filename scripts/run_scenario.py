#!/usr/bin/env python3
# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import pathlib
import shutil
import struct
import subprocess  # nosec B404

import numpy as np
import safetensors
import torch

from ng_model_gym.core.utils import dds_utils


def generate_preprocess_push_constants(
    tensors,
    frame_idx,
    parameters_json,
    input_width,
    input_height,
    output_width,
    output_height,
):
    """Generates a bytearray containing the preprocess push constants for a given frame index"""

    buffer = bytearray(128)

    # layout(offset =  0)  float4  _DeviceToViewDepth;   //  16 B
    device_to_view_depth = tensors.get_slice("depth_params")[frame_idx]
    struct.pack_into("<ffff", buffer, 0, *device_to_view_depth)

    # layout(offset = 16)  float4  _JitterOffset;        //  16 B (.xy = pixels, .zw = uvs)
    jitter_offset = tensors.get_slice("jitter")[frame_idx]
    struct.pack_into(
        "<ffff",
        buffer,
        16,
        jitter_offset[1],  # Safetensors has YX, scenario needs XY
        jitter_offset[0],
        jitter_offset[1] / input_width,
        jitter_offset[0] / input_height,
    )

    # layout(offset = 32)  float4  _JitterOffsetTm1;     //  16 B (.xy = pixels, .zw = uvs)
    jitter_offset_tm1 = tensors.get_slice("jitter")[max(frame_idx - 1, 0)]
    struct.pack_into(
        "<ffff",
        buffer,
        32,
        jitter_offset_tm1[1],  # Safetensors has YX, scenario needs XY
        jitter_offset_tm1[0],
        jitter_offset_tm1[1] / input_width,
        jitter_offset_tm1[0] / input_height,
    )

    # layout(offset = 48)  float4  _ScaleFactor;         //  16 B (.xy = scale, .zw = inv scale)
    struct.pack_into(
        "<ffff",
        buffer,
        48,
        output_width / input_width,
        output_height / input_height,
        input_width / output_width,
        input_height / output_height,
    )

    # layout(offset = 64)  int32_t2 _OutputDims;         //   8 B
    struct.pack_into("<ii", buffer, 64, output_width, output_height)

    # layout(offset = 72)  int32_t2 _InputDims;          //   8 B
    struct.pack_into("<ii", buffer, 72, input_width, input_height)

    # layout(offset = 80)  float2   _InvOutputDims;      //   8 B
    struct.pack_into("<ff", buffer, 80, 1.0 / output_width, 1.0 / output_height)

    # layout(offset = 88)  float2   _InvInputDims;       //   8 B
    struct.pack_into("<ff", buffer, 88, 1.0 / input_width, 1.0 / input_height)

    # layout(offset = 96)  half4    _QuantParams;        //   8 B  (.xy SINT, .zw SNORM)
    struct.pack_into(
        "<eeee",
        buffer,
        96,
        # Note inverse as this is used for quantization, not dequantization
        1.0 / parameters_json["inputs"]["x"]["SINT"]["scale"],
        parameters_json["inputs"]["x"]["SINT"]["zero_point"],
        parameters_json["outputs"]["activation_post_process_70"]["SNORM"]["scale"],
        parameters_json["outputs"]["activation_post_process_70"]["SNORM"]["zero_point"],
    )

    # layout(offset = 104) half4    _MotionDisThreshPad;
    #   //   8 B  (.xyzw = motion/disocclusion thresholds)
    struct.pack_into(
        "<eeee",
        buffer,
        104,
        0.1**2.0,
        0.5**2.0,
        parameters_json["learnt_constants"]["dm_scale"],
        0.0,
    )

    # layout(offset = 112) half2    _Exposure;
    #   //   4 B  (.x = exposure, .y = 1/exp)
    # Even though the safetensors file contains an exposure value, this is ignored
    # in training and evaluation so we also ignore it here and use the same override.
    exposure_override = 7.38905609
    struct.pack_into("<ee", buffer, 112, exposure_override, 1.0 / exposure_override)

    # layout(offset = 116) half2    _HistoryPad;         //   4 B
    struct.pack_into("<ee", buffer, 116, 0.0 if frame_idx == 0 else 1.0, 0.0)

    # layout(offset = 120) int32_t2 _Padding;            //   8 B

    return buffer


def compute_jitter_tile_offset(jitter, scale, idx_mod):
    """
    Computes LUT tile offset (dx, dy) for a given jitter and non-uniform scale and tiling.

    Args:
        jitter (Tuple[float, float]): (jx, jy) jitter offset
        scale (Tuple[float, float]): (sx, sy) scale factor (X and Y)
        idx_mod (Tuple[int, int]): (mx, my) LUT tiling modulus (X and Y)

    Returns:
        Tuple[int, int]: (dx_offset, dy_offset)
    """
    jx, jy = jitter
    sx, sy = scale
    mx, my = idx_mod

    # Project base and jittered LR pixel centers to HR index space
    base_hr_x = int((0.5 * sx) // 1)
    base_hr_y = int((0.5 * sy) // 1)

    jittered_hr_x = int(((jx + 0.5) * sx) // 1)
    jittered_hr_y = int(((jy + 0.5) * sy) // 1)

    dx_offset = (jittered_hr_x - base_hr_x) % mx
    dy_offset = (jittered_hr_y - base_hr_y) % my

    return dx_offset, dy_offset


def generate_postprocess_push_constants(
    tensors,
    frame_idx,
    parameters_json,
    input_width,
    input_height,
    output_width,
    output_height,
):
    """Generates a bytearray containing the postprocess push constants for a given frame index"""
    buffer = bytearray(76)

    # layout(offset =  0) int32_t2 _OutputDims;        //  8 B
    struct.pack_into("<ii", buffer, 0, output_width, output_height)

    # layout(offset =  8) int32_t2 _InputDims;         //  8 B
    struct.pack_into("<ii", buffer, 8, input_width, input_height)

    # layout(offset = 16) float2   _InvOutputDims;     //  8 B
    struct.pack_into("<ff", buffer, 16, 1.0 / output_width, 1.0 / output_height)

    # layout(offset = 24) float2   _InvInputDims;      //  8 B
    struct.pack_into("<ff", buffer, 24, 1.0 / input_width, 1.0 / input_height)

    # layout(offset = 32) float2   _Scale;             //  8 B
    struct.pack_into(
        "<ff", buffer, 32, output_width / input_width, output_height / input_height
    )

    # layout(offset = 40) float2   _InvScale;          //  8 B
    struct.pack_into(
        "<ff", buffer, 40, input_width / output_width, input_height / output_height
    )

    # layout(offset = 48) int16_t2 _IndexModulo;       //  4 B
    struct.pack_into("<hh", buffer, 48, 2, 2)

    # layout(offset = 52) half2    _QuantParams;       //  4 B
    struct.pack_into(
        "<ee",
        buffer,
        52,
        parameters_json["outputs"]["activation_post_process_45"]["SNORM"]["scale"],
        parameters_json["outputs"]["activation_post_process_45"]["SNORM"]["zero_point"],
    )

    # layout(offset = 56) int16_t2 _LutOffset;         //  4 B
    jitter_offset = tensors.get_slice("jitter")[frame_idx]
    (lut_offset_x, lut_offset_y) = compute_jitter_tile_offset(
        (jitter_offset[1], jitter_offset[0]),  # Note that safetensors stores YX
        (output_width / input_width, output_height / input_height),
        (2, 2),
    )
    struct.pack_into("<hh", buffer, 56, lut_offset_x, lut_offset_y)

    # layout(offset = 60) half2    _ExposurePair;      //  4 B
    # Even though the safetensors file contains an exposure value, this is ignored in
    # training and evaluation so we also ignore it here and use the same override.
    exposure_override = 7.38905609
    struct.pack_into("<ee", buffer, 60, exposure_override, 1.0 / exposure_override)

    # layout(offset = 64) half2    _HistoryPad;        //  4 B
    struct.pack_into("<ee", buffer, 64, 0.0 if frame_idx == 0 else 1.0, 0.0)

    # layout(offset = 68) half2    _MotionThreshPad;   //  4 B (.x = motion, .y = unused)
    struct.pack_into("<ee", buffer, 68, 0.1**2.0, 0.0)

    # layout(offset = 72) int32_t  _Padding0;          //  4 B (explicit pad for alignment)

    return buffer


def write_scenario(
    tensors, frame_idx, parameters_json, base_scenario_path, dest_scenario_path
):
    """
    Writes out a scenario description for a given frame index,
    using the given base scenario as a starting point.
    """

    # Clean up any existing scenario folder
    if os.path.exists(dest_scenario_path):
        shutil.rmtree(dest_scenario_path)
    os.makedirs(dest_scenario_path)

    # Get input and output sizes, including padding
    unpadded_input_width = tensors.get_slice("render_size")[frame_idx][1].item()
    unpadded_input_height = tensors.get_slice("render_size")[frame_idx][0].item()
    if unpadded_input_width != 960 or unpadded_input_height != 540:
        # To support other sizes we'd need to update the padding below
        raise ValueError(
            "Only 960x540 input resolution is supported, got "
            + f"{unpadded_input_width}x{unpadded_input_height}"
        )
    input_pad_w, input_pad_h = (0, 4)
    input_width = unpadded_input_width + input_pad_w
    input_height = unpadded_input_height + input_pad_h

    unpadded_output_width = tensors.get_slice("outDims")[frame_idx][1].item()
    unpadded_output_height = tensors.get_slice("outDims")[frame_idx][0].item()
    if unpadded_output_width != 1920 or unpadded_output_height != 1080:
        # To support other sizes we'd need to update the padding below
        raise ValueError(
            "Only 1920x1080 output resolution is supported, got "
            + f"{unpadded_output_width}x{unpadded_output_height}"
        )
    output_pad_w, output_pad_h = (0, 8)
    output_width = unpadded_output_width + output_pad_w
    output_height = unpadded_output_height + output_pad_h

    # Copy files from the base scenario that we leave unaltered
    shutil.copy(base_scenario_path / "0_pre_process.spv", dest_scenario_path)
    shutil.copy(base_scenario_path / "1_nss.vgf", dest_scenario_path)
    shutil.copy(base_scenario_path / "2_post_process.spv", dest_scenario_path)
    shutil.copy(base_scenario_path / "scenario.json", dest_scenario_path)

    # Generate and save new preprocess push constants
    preprocess_push_constants = generate_preprocess_push_constants(
        tensors,
        frame_idx,
        parameters_json,
        input_width,
        input_height,
        output_width,
        output_height,
    )
    np.save(
        dest_scenario_path / "0_pre_process_push_consts.npy",
        np.frombuffer(preprocess_push_constants, dtype=np.byte),
    )

    # Generate and save new postprocess push constants
    postprocess_push_constants = generate_postprocess_push_constants(
        tensors,
        frame_idx,
        parameters_json,
        input_width,
        input_height,
        output_width,
        output_height,
    )
    np.save(
        dest_scenario_path / "2_post_process_push_consts.npy",
        np.frombuffer(postprocess_push_constants, dtype=np.byte),
    )

    # Write DDS files for the texture inputs, adding mirror padding where necessary
    apply_padding = lambda x: torch.nn.functional.pad(
        x, (0, input_pad_w, 0, input_pad_h), mode="reflect"
    )

    in_colour = tensors.get_slice("colour_linear")[frame_idx]
    in_colour = apply_padding(in_colour)
    dds_utils.save_dds(
        in_colour.numpy(),
        dest_scenario_path / "in_colour.dds",
        dds_utils.DXGI_FORMAT_R11G11B10_FLOAT,
    )

    in_depth = tensors.get_slice("depth")[frame_idx]
    in_depth = apply_padding(in_depth)
    dds_utils.save_dds(
        in_depth.numpy(),
        dest_scenario_path / "in_depth.dds",
        dds_utils.DXGI_FORMAT_R32_FLOAT,
    )

    in_motion = tensors.get_slice("motion_lr")[frame_idx]
    # Safetensors has channel order YX and stores in units of pixels.
    # Scenario runner requires XY order and units of normalized UVs
    in_motion = in_motion.flip(dims=[0])
    # Note UNPADDED input width/height here
    scale = torch.tensor([unpadded_input_width, unpadded_input_height]).resize(2, 1, 1)
    in_motion = in_motion / scale
    in_motion = apply_padding(in_motion)
    dds_utils.save_dds(
        in_motion.numpy(),
        dest_scenario_path / "in_motion.dds",
        dds_utils.DXGI_FORMAT_R16G16_FLOAT,
    )

    # Link to DDS files produced from the previous frame (using symlinks to avoid copies)
    # The first frame needs special handling as there is no previous frame.
    # This logic should match that in _init_history_buffers of model_v1.py
    if frame_idx == 0:
        os.symlink("in_depth.dds", dest_scenario_path / "in_depth_tm1.dds")

        in_derivative_tm1 = torch.zeros(
            (2, input_height, input_width), dtype=torch.uint8
        )
        dds_utils.save_dds(
            in_derivative_tm1.numpy(),
            dest_scenario_path / "in_derivative_tm1.dds",
            dds_utils.DXGI_FORMAT_R8G8_UNORM,
        )

        in_feedback_tm1 = (
            torch.ones((4, input_height, input_width), dtype=torch.int8) * -127
        )
        dds_utils.save_dds(
            in_feedback_tm1.numpy(),
            dest_scenario_path / "in_feedback_tm1.dds",
            dds_utils.DXGI_FORMAT_R8G8B8A8_SNORM,
        )

        in_history = torch.zeros((3, output_height, output_width), dtype=torch.float16)
        dds_utils.save_dds(
            in_history.numpy(),
            dest_scenario_path / "in_history.dds",
            dds_utils.DXGI_FORMAT_R11G11B10_FLOAT,
        )

        in_history_nearest_offset_tm1 = torch.zeros(
            (1, input_height, input_width), dtype=torch.uint8
        )
        dds_utils.save_dds(
            in_history_nearest_offset_tm1.numpy(),
            dest_scenario_path / "in_nearest_offset_tm1.dds",
            dds_utils.DXGI_FORMAT_R8_UNORM,
        )
    else:
        os.symlink(
            os.path.join("..", f"frame_{(frame_idx-1):04d}", "in_depth.dds"),
            dest_scenario_path / "in_depth_tm1.dds",
        )
        os.symlink(
            os.path.join("..", f"frame_{(frame_idx-1):04d}", "out_derivative.dds"),
            dest_scenario_path / "in_derivative_tm1.dds",
        )
        os.symlink(
            os.path.join("..", f"frame_{(frame_idx-1):04d}", "out_feedback.dds"),
            dest_scenario_path / "in_feedback_tm1.dds",
        )
        os.symlink(
            os.path.join("..", f"frame_{(frame_idx-1):04d}", "out_colour.dds"),
            dest_scenario_path / "in_history.dds",
        )
        os.symlink(
            os.path.join("..", f"frame_{(frame_idx-1):04d}", "out_nearest_offset.dds"),
            dest_scenario_path / "in_nearest_offset_tm1.dds",
        )


def main():
    """Main function"""

    p = argparse.ArgumentParser(
        description="Uses the ML SDK Scenario Runner to evaluate upscaled frames for all frames "
        "in a given .safetensor file"
    )
    p.add_argument(
        "--model-repo",
        type=pathlib.Path,
        required=True,
        help="Path to the model repository (containing the base scenario description). "
        "Known working revision: 6d5a8d4ad0792aba40e8a02fc84e5a483d651a97",
    )
    p.add_argument(
        "--input-safetensors",
        required=True,
        help="Path to the .safetensors file containing the frames to evaluate",
    )
    p.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Path to a folder where results will be written",
    )
    args = p.parse_args()

    # Read parameters.json which includes quantization params and other metadata
    with open(
        os.path.join(args.model_repo, "scenario", "parameters.json"),
        "r",
        encoding="utf-8",
    ) as f:
        parameters_json = json.load(f)

    # Read the safetensors file manually rather than using our dataloader class, as that brings in
    # dependencies that we don't want for this simple script
    with safetensors.safe_open(args.input_safetensors, framework="pt") as tensors:
        num_frames = int(tensors.metadata()["Length"])

        for frame_idx in range(num_frames):
            print(f"Frame {frame_idx}/{num_frames}...")

            # Create a scenario description for this frame
            base_scenario_path = args.model_repo / "scenario"
            frame_scenario_path = args.output_dir / f"frame_{frame_idx:04d}"
            write_scenario(
                tensors,
                frame_idx,
                parameters_json,
                base_scenario_path,
                frame_scenario_path,
            )

            # Launch scenario-runner to run the scenario and produce the output files
            subprocess.run(  # nosec B603
                args=[
                    args.model_repo / "bin" / "windows-x86_64" / "scenario-runner.exe",
                    "--log-level",
                    "warning",
                    "--scenario",
                    frame_scenario_path / "scenario.json",
                    "--output",
                    frame_scenario_path,  # Place all the output files in the same folder
                ],
                # Modify the environment so that the emulation layers are loaded
                env={
                    **os.environ,
                    "VK_LAYER_PATH": str(args.model_repo / "bin" / "windows-x86_64"),
                    "VK_INSTANCE_LAYERS": (
                        "VK_LAYER_ML_Graph_Emulation;VK_LAYER_ML_Tensor_Emulation"
                    ),
                },
                check=True,
            )


if __name__ == "__main__":
    main()
