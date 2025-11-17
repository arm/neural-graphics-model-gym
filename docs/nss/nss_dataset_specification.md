<!---
SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->

# NSS Dataset Specification

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset contents](#dataset-contents)
3. [Capture layout](#capture-layout)
    * [Images](#images)
    * [Metadata](#metadata)
      * [Per-sequence](#per-sequence)
      * [Per-frame](#per-frame)

## Introduction

This document describes the specification for data that needs to be captured from your game engine of choice to train Neural Super Sampling (NSS).
Python [scripts](../../scripts/safetensors_generator/safetensors_writer.py) included within this repository expect your engine captured data to be in the format described by this document.

## Dataset contents

A dataset comprises a series of ground truth images, which are high-quality images captured at the output target resolution we wish to upscale towards, e.g., 1080p, and input images captured at a variety of resolutions to satisfy different upscaling ratios, e.g., for a `ratio` of 2x2, and a ground truth resolution of 1080p, the input images will be captured at 540p.

In addition to images, the dataset also comprises metadata, which captures the state of the engine at the time that the images were rendered.

## Capture layout
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
├── motion_gt/
│   ├── 0000/
│   │   ├── 0000.exr
│   │   ├── 0001.exr
│   │   └── ...
│   ├── 0001/
│   │   ├── 0000.exr
│   │   ├── 0001.exr
│   │   └── ...
│   └── ...
└── x{ratio}/
    ├── color/
    │   ├── 0000/
    │   │   ├── 0000.exr
    │   │   ├── 0001.exr
    │   │   └── ...
    │   ├── 0001/
    │   │   ├── 0000.exr
    │   │   ├── 0001.exr
    │   │   └── ...
    │   └── ...
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
    └── motion/
        ├── 0000/
        │   ├── 0000.exr
        │   ├── 0001.exr
        │   └── ...
        ├── 0001/
        │   ├── 0000.exr
        │   ├── 0001.exr
        │   └── ...
        └── ...
```

At the root of the dataset you will find the `ground_truth` images, several directories in a per-ratio basis and a JSON file containing metadata for each sequence. Inside these per-ratio directories, each folder will contain a succession of subdirectories based on how many sequences are contained in the dataset, each will contain the generated images in a `.exr` file.

### Images

| Image    | Format | Resolution                | Description | Requirement |
| -------- | ------- |---------------------------| ------- | ------- |
|`ground_truth`|`R16G16B16A16_SFLOAT`| `1920 x 1080`             | Color target rendered natively at 8k, then downsampled to 1080p with a box filter. Stored in linear HDR. | Mandatory |
|`color`|`R16G16B16A16_SFLOAT`| `1920/ratio x 1080/ratio` | Jittered, low resolution color texture before upscaling, stored in linear HDR. | Mandatory |
|`depth`|`R32_SFLOAT` | `1920/ratio x 1080/ratio` | Jittered, low resolution depth texture before upscaling. Preferably forward-Z. | Mandatory |
|`motion`| `R16G16_SFLOAT`| `1920/ratio x 1080/ratio` | Non-Jittered, low resolution velocity texture containing motion information from the frame's velocity texture, stored in UV space.| Mandatory |
|`motion_gt`| `R16G16_SFLOAT`| `1920 x 1080`             | Non-Jittered, output resolution velocity texture containing motion information from the frame's velocity texture, stored in UV space.| Mandatory |

### Metadata

Metadata is split up into two parts "per sequence," which reflects global information that is uniform for all frames within the contiguous sequence, and "per frame" which is information that varies between contiguous frames.

#### Per-sequence

| Metadata    | DTYPE  | Description | Requirement |
| -------- | ------- | ------- | ------- |
|`Version` | `str` | The version of the specification to which the dataset conforms. | Mandatory |
|`EmulatedFramerate` | `int` | The frame rate that the dataset collection tool emulated during capture, e.g., 60. | Mandatory |
|`TargetResolution` | `dict` | An object containing the horizontal and vertical ground truth target resolution e.g., `{"X": 1920,"Y": 1080}` for 1080p. | Mandatory |
|`Samples` | `dict` | An object containing information pertaining to the sampling strategy, number of unique positions, and whether the positions have been quantized when the jittered frame content was captured. e.g., `{"Sequence": "TiledHalton", "Count": 16, "Quantized": true}`. | Mandatory |
|`Frames` | `list` | A list containing the "per-frame" metadata. | Mandatory |
|`ReverseZ` | `bool` | Whether the depth is stored as reversed-Z order. Preferably this should be `false` i.e. forward-Z. | Mandatory |
|`OpaqueOnlyColor_Exported`| `bool` | Whether opaque-only color textures are generated. Used for generating the reactive mask. | Optional |
|`UnjitteredSrcColor_Exported` | `bool` | Whether the unjittered (resampled) color texture has been generated. Useful for debugging jitter vectors. | Optional |
|`UpscalingRatiosIndices`| `dict` | A mapping between upscaling ratio descriptions and the indices used to look up entries in the `NormalizedPerRatioJitter` list for each frame. | Optional |
|`MotionVectorsDilated` | `bool` | Whether the motion vectors have been pre-dilated by the capture application. | Optional |

#### Per-frame

| Metadata    | DTYPE  | Description | Requirement |
| -------- | ------- | ------- | ------- |
|`Frame` | `int` | Frame index. | Mandatory |
|`FovX` | `float` | Horizontal Fov of the camera (radians). | Mandatory |
|`FovY` | `float` | Vertical Fov of the camera (radians). | Mandatory |
|`CameraNearPlane` | `float` | Near plane of the camera in meters. | Mandatory |
|`CameraFarPlane` | `float` | Far plane in meters from the camera. `-1.0` is used to indicate an infinite far plane.| Mandatory |
|`ViewProjection` | `list` | ViewProjection matrix's raw data (`float`, column major). | Mandatory |
|`Jitter` | `dict` | The frame's raw jitter offset in pixels. e.g., `{"X": 0.25, "Y": 0.67}`. | Mandatory |
|`NormalizedPerRatioJitter` | `list` | List containing per-upscaling ratio normalized jitter offsets (i.e. NormalizedPerRatioJitter [x2_index] would be the jitter offset used for decimating the textures in the x2 scenario). Normalized jitter values are the `Jitter {-0.5, 0.5}` divided by the SrcRes (i.e 920x540) | Mandatory |
|`Exposure` | `float` | Exposure value used in the frame for color correction. | Optional |
