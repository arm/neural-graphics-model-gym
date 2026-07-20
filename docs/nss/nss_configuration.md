<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->
# How to configure NSS models

Before reading further, please read `docs/usage.md`. That file explains how to configure and use Model Gym.


## Introduction

Model Gym's NSS model requires a configuration file, typically named `nss_config.json`. This file is initially generated using the following command:

    ng-model-gym init nss

This document is a guide to the most common changes you will need to make to `nss_config.json`. For more details, refer to `schema_config.json`, which contains a detailed description of every configuration setting.


## Data you must provide

It's essential to provide the following information:

- `dataset` → `path` → `train`: path to a directory containing training data in the form of a collection of `.safetensors` files.
- `dataset` → `path` → `validation`: path to a directory containing validation data.
- `dataset` → `path` → `test`: path to a directory containing test data.


## Other settings you may wish to change

- `model` → `quality`: Specifies a quality level: a specific compromise between speed and output quality. Quality levels `high`, `mid` and `low` are defined in `src/ng_model_gym/usecases/nss/model/quality_modes.py`. These have the following meanings:

  | Level    | Quality  | Runtime cost | Pre-process resolution | Depth scatter resolution | Post-process color filtering | Temporal history sampling | Kernel Prediction Network (KPN) size |
  |----------|----------|--------------|------------------------|--------------------------|------------------------------|---------------------------|--------------------------------------|
  | `high`   | Highest  | Highest      | Full                   | Half                     | Full                         | Catmull-Rom               | 6×6                                  |
  | `mid`    | Balanced | Balanced     | Half                   | Quarter                  | Sparse                       | Catmull-Rom               | 4×4                                  |
  | `low`    | Lowest   | Lowest       | Half                   | Quarter                  | Sparse                       | Bilinear                  | 4×4                                  |

  In everyday language:

  - "high": highest runtime cost but best-quality option. Checks the current frame and depth/motion detail more thoroughly, uses a larger image filter for clean-up, and samples previous frames more accurately.
  - "low": lowest runtime cost but lowest-quality option. Uses lighter current-frame and depth/motion checks, and samples previous frames less accurately. May show more flicker or motion artifacts around fine detail and moving objects.
  - "mid": balanced option. Similar to "low" but samples previous frames more accurately, like "high".

- `model` → `recurrent_samples`: Number of consecutive video frames the model trains on at once. A higher setting may improve temporal stability at a cost of greater memory usage and running more slowly.

- `model` → `gt_history_augmentation`: `true` if you wish the first frame in a recurrent training window to occasionally be based on ground truth. This can help the model learn to use "good" history information rather than simply training it to recover from an empty reset. `model` → `gt_history_augmentation_chance` sets the percentage probability of ground truth being used.

- `dataset` → `exposure`: Logarithmic exposure override, to brighten training images before tonemapping takes place. For example, `2` multiplies brightness by `e²`. A null or absent value takes exposure values from the dataset.

- `dataset` → `tonemapper`: Selects the tonemapping method:

  - `reinhard`: Simple, robust HDR compression that reduces each color channel independently; a good default when you want predictable brightness control.
  - `karis`: Compresses brightness using the strongest RGB channel. Tends to preserve color balance better in bright highlights.
  - `log`: Applies natural-log compression, strongly reducing bright values while keeping darker detail relatively visible.
  - `log10`: Applies base-10 log compression. Similar to `log` but produces lower brightness values.
  - `log_norm`: Applies log compression normalized to the expected HDR range, useful when you want values mapped into a more consistent bounded scale.
  - `aces`: Uses a filmic ACES-style curve with smooth highlight roll-off, useful when perceptual image appearance matters more than simple numeric compression.
