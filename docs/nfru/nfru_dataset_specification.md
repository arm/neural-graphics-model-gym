<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->

# NFRU Dataset Specification

## Table of Contents

1. [Introduction](#introduction)
2. [Terminology](#terminology)
3. [Dataset contents](#dataset-contents)
    * [File hierarchy](#file-hierarchy)
    * [Images](#images)
    * [Metadata](#metadata)
      * [Per-capture](#per-capture)
      * [Per-frame](#per-frame)

## Introduction

This document specifies data that must be captured from your game engine of choice if you wish to train a model implementing Neural Frame Rate Upscaling (NFRU).

A [script included within this repository](../../scripts/safetensors_generator/safetensors_writer.py) transforms this NFRU data into a Safetensors file which can be used by Model Gym's training feature.

## Terminology

- **Sequence** – a contiguous range of frames within a capture that contains no camera cuts.
- **Capture** – a chronological recording of frames exported from the engine. A single capture may include multiple camera cuts. This will have an index such as `0000`.
- **Dataset** – a collection of captures grouped for training or evaluation.

## Dataset contents

The Neural Frame Rate Upscaling (NFRU) dataset has been designed assuming Neural Super Sampling (NSS) is also being used. Therefore, this dataset aligns closely to the NSS specification. Input/output resolutions of content map directly to NSS inputs and outputs, the key differences being:

- Ground truth NSS frames are used as input NFRU frames
- Extra motion vectors need to be provided to match low frame rate input (see directory `motion_m2` below)
- We assume the dataset is captured at 60FPS (using 30 → 60 upscaling)

### File hierarchy

At the root of the dataset, you will find:

- `0000.json`, `0001.json`, etc.: JSON files containing metadata for each numbered capture. This captures the state of the engine at the time that the images were rendered.
- `ground_truth`: Directory containing "ground truth" images. These are high-quality images captured at the output target resolution, for example, 1080p.
- `x2`: Directories containing images, etc. captured at a 2x (50%) resolution. For example, for a ground truth resolution of 1080p, the input images in this directory should be captured at 540p. These directories contains `depth`, `motion_m1` and `motion_m2` subdirectories (see "images" section below for more details).

`ground_truth`, `x2/depth`, `x2/motion_m1` and `x2/motion_m2` all contain a single subdirectory for each capture contained in the dataset. Each per-capture subdirectory contains numbered `.exr` images representing the frames making up the capture. `x2/motion_m2` only contains frame for even indexes as it represents motion across two 60 FPS frames.

```
<DATASET PATH>/
├── 0000.json
├── 0001.json
├── ...
├── ground_truth/
│   ├── 0000/
│   │   ├── 0000.exr
│   │   ├── 0001.exr
│   │   └── ...
│   ├── 0001/
│   │   ├── 0000.exr
│   │   ├── 0001.exr
│   │   └── ...
│   └── ...
└── x2/
    ├── depth/
    │   ├── 0000/
    │   │   ├── 0000.exr
    │   │   ├── 0001.exr
    │   │   └── ...
    │   ├── 0001/
    │   │   ├── 0000.exr
    │   │   ├── 0001.exr
    │   │   └── ...
    │   └── ...
    ├── motion_m1/
    │   ├── 0000/
    │   │   ├── 0000.exr
    │   │   ├── 0001.exr
    │   │   └── ...
    │   ├── 0001/
    │   │   ├── 0000.exr
    │   │   ├── 0001.exr
    │   │   └── ...
    │   └── ...
    └── motion_m2/
        ├── 0000/
        │   ├── 0000.exr
        │   ├── 0002.exr
        │   └── ...
        ├── 0001/
        │   ├── 0000.exr
        │   ├── 0002.exr
        │   └── ...
        └── ...
```

### Images

| Image | Format | Resolution | Description | Requirement |
| ----- | ------ | ---------- | ----------- | ----------- |
|`ground_truth` | `R16G16B16A16_SFLOAT` (`A` channel is ignored) | `1920 x 1080` | Color target rendered natively at 8k, then downsampled to 1080p with a box filter. Stored in linear HDR. | Mandatory |
|`x2/depth` | `R32_SFLOAT` | `960 x 540` | Jittered, low resolution depth texture before upscaling. Preferably forward-Z. | Mandatory |
|`x2/motion_m1` |`R16G16B16A16_SFLOAT` (`B` and `A` channels are ignored) | `960 x 540` | Non-jittered, low-resolution velocity texture containing motion information from the frame's velocity texture, stored in UV space. | Mandatory |
|`x2/motion_m2` |`R16G16B16A16_SFLOAT` (`B` and `A` channels are ignored) | `960 x 540` | Non-jittered, low-resolution velocity texture containing motion from the current frame to frame T-2 (skip motion vector), stored in UV space. Even frame indices only. Indexing must be correctly applied if not capturing interleaved 30 FPS data (see “a note on indexing”) | Mandatory |

### A note on indexing

Indexing is respective to the higher capture frame rate (that is: 60 FPS). Care must be taken to ensure that indexing and motion vectors align correctly. For example, for `motion_m2` vectors, the following is required at minimum:

- `0000.exr`
- `0002.exr`
- `0004.exr`
- `0006.exr`

This corresponds to 30 FPS vectors, but is aligned correctly to 60 FPS indexing.

NFRU interpolates between two frames (T-1 and T+1) to generate an intermediate frame at T. As a result, the first and last frames cannot be interpolated. It is important to ensure that the results directories contain only valid interpolations and are properly aligned with the corresponding ground truth. In other words, the NFRU results should never include `0000.exr`, since this does not represent a valid interpolation.

### Metadata

Metadata is split into two parts:

- "Per-capture": Metadata that is uniform for all frames within a contiguous capture.
- "Per-frame": Metadata that changes frame-by-frame.

#### Per-capture

| Metadata | DTYPE | Description | Requirement |
| -------- | ----- | ----------- | ----------- |
|`Version` | `str` | Version of the specification to which the dataset conforms. | Mandatory |
|`EmulatedFramerate` | `int` | Frame rate that the dataset collection tool emulated during capture. For example: 60. | Mandatory |
|`TargetResolution` | `dict` | Object containing the horizontal and vertical ground truth target resolution. For example: `{"X": 1920,"Y": 1080}` for 1080p. | Mandatory |
|`Samples` | `dict` | Object containing information pertaining to the sampling strategy, number of unique positions, and whether the positions have been quantized when the jittered frame content was captured. For example: `{"Sequence": "TiledHalton", "Count": 16, "Quantized": true}`. | Mandatory |
|`Frames` | `list` | List containing per-frame metadata. | Mandatory |
|`ReverseZ` | `bool` | Whether depth is stored in reversed-Z order. Preferably this should be `false` – that is, forward-Z. | Mandatory |
|`UpscalingRatiosIndices`| `dict` | Contains a single member, `x2_index`, which must be `0`. | Mandatory |
|`OpaqueOnlyColor_Exported`| `bool` | Whether opaque-only color textures are generated. Used for generating the reactive mask. | Optional |
|`UnjitteredSrcColor_Exported` | `bool` | Whether the unjittered (resampled) color texture has been generated. Useful for debugging jitter vectors. | Optional |
|`MotionVectorsDilated` | `bool` | Whether the motion vectors have been pre-dilated by the capture application. | Optional |
|`InverseY` | `bool` | Whether view-projection Y inversion fix should be applied by the reader. | Optional |
|`NearPlane` | `float` | Near plane of the camera, in meters. Fallback used if a per-frame near plane is not provided. | Optional |
|`FarPlane` | `float` | Far plane from the camera, in meters. Fallback used if a per-frame far plane is not provided. `-1.0` indicates infinite far plane. | Optional |

#### Per-frame

| Metadata | DTYPE | Description | Requirement |
| -------- | ----- | ----------- | ----------- |
|`Frame` | `int` | Frame index. | Mandatory |
|`FovX` | `float` | Horizontal field of view of the camera, in radians. | Mandatory |
|`FovY` | `float` | Vertical field of view of the camera, in radians. | Mandatory |
|`CameraNearPlane` | `float` | Near plane of the camera, in meters. | Optional |
|`CameraFarPlane` | `float` | Far plane in meters from the camera. `-1.0` indicates an infinite far plane. | Optional |
|`CameraPosition` | `dict` | Camera position in world space. For example: `{"X": -20863.1640625, "Y": -15147.1884765625, "Z": 2709.897705078125}`. | Mandatory |
|`CameraUp` | `dict` | Camera up normalized vector in world space. For example: `{"X": -0.0049013220705091953, "Y": 0.1739216148853302, "Z": 0.98474729061126709}`. | Mandatory |
|`CameraRight` | `dict` | Camera right normalized vector in world space. For example: `{"X": -0.99960315227508545, "Y": -0.028170028701424599, "Z": -1.7208456881689926e-15}`. | Mandatory |
|`CameraForward` | `dict` | Camera forward normalized vector in world space. For example: `{"X": -0.027740359306335449, "Y": 0.98435652256011963, "Z": -0.17399066686630249}`. | Mandatory |
|`View` | `list` | View matrix raw data (`float`, 16 values). | Optional |
|`Projection` | `list` | Projection matrix raw data (`float`, 16 values). | Optional |
|`ViewProjection` | `list` | View projection matrix's raw data (`float`, row-major, 16 values). | Mandatory |
|`Jitter` | `dict` | Raw jitter offset in pixels. For example: `{"X": 0.25, "Y": 0.67}`. | Mandatory |
|`NormalizedPerRatioJitter` | `list` | Single-element list containing normalized jitter offsets. Can be calculated as `Jitter` divided by the source resolution (960 x 540). For example, the `Jitter` value above produces a `NormalizedPerRatioJitter` value of `{"X": 0.00026041666666667, "Y": 0.00069791666666667}`. | Mandatory |
|`Exposure` | `float` | Final exposure value used for color correction. | Mandatory |
