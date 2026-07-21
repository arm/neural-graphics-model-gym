<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->
# How to configure NSS models

Before reading further, see the [usage guide](../usage.md). It explains how to
configure and run Model Gym.


## Introduction

Model Gym's NSS model requires a JSON configuration file. Generate one with the
following command:

    ng-model-gym init nss-v1

This document is a guide to the most common configuration changes. For every
available setting, refer to the generated
`schema_config.json` file or run `ng-model-gym config-options`.


## Dataset paths

Provide dataset paths for your required workflow:

- `dataset` → `path` → `train`: path to the training data.
- `dataset` → `path` → `validation`: path to the validation data, if validation is enabled.
- `dataset` → `path` → `test`: path to the test data, if evaluation is enabled.


## Other settings you may wish to change

### Scale

- `model` → `scale`: Upscaling factor used by NSS v1 training and evaluation.
  It defaults to `2.0` and must be greater than `1.0`.

### Quality and recurrent training

- `model` → `quality`: Specifies a compromise between speed and output
  quality. The available levels have the following meanings:

  | Level    | Quality  | Runtime cost | Pre-process resolution | Depth scatter resolution | Post-process color filtering | Temporal history sampling | Kernel Prediction Network (KPN) size |
  |----------|----------|--------------|------------------------|--------------------------|------------------------------|---------------------------|--------------------------------------|
  | `high`   | Highest  | Highest      | Full                   | Half                     | Full                         | Catmull-Rom               | 6×6                                  |
  | `mid`    | Balanced | Balanced     | Half                   | Quarter                  | Sparse                       | Catmull-Rom               | 4×4                                  |
  | `low`    | Lowest   | Lowest       | Half                   | Quarter                  | Sparse                       | Bilinear                  | 4×4                                  |

  In everyday language:

  - "high": highest runtime cost but best-quality option. Checks the current frame and depth/motion detail more thoroughly, uses a larger image filter for clean-up, and samples previous frames more accurately.
  - "low": lowest runtime cost but lowest-quality option. Uses lighter current-frame and depth/motion checks, and samples previous frames less accurately. May show more flicker or motion artifacts around fine detail and moving objects.
  - "mid": balanced option. Similar to "low" but samples previous frames more accurately, like "high".

- `model` → `recurrent_samples`: Number of consecutive video frames the model trains on at once. A higher setting may improve temporal stability at a cost of greater memory usage and training more slowly.

- `model` → `gt_history_augmentation`: `true` if you wish the first frame in a recurrent training window to occasionally be based on ground truth. This can help the model learn to use "good" history information rather than simply training it to recover from an empty reset. `model` → `gt_history_augmentation_chance` sets the percentage probability of ground truth being used.

### Checkpoint quality compatibility

When fine-tuning, evaluating, or exporting, checkpoints can be used with these
configured quality levels:

| Checkpoint quality | `high` model | `mid` model | `low` model |
|--------------------|--------------|-------------|-------------|
| `high`             | Yes          | Yes         | Yes         |
| `mid`              | No           | Yes         | Yes         |
| `low`              | No           | Yes         | Yes         |

Any required adjustment from `high` to `mid` or `low` is automatic.

Use `--finetune` to start a new training run from checkpoint weights. This is
the right option when changing from `high` quality to `mid` or `low`.

Use `--resume` to continue an existing training run. Keep the same quality
setting and training configuration because resume restores the full training
state. See the usage guide for the
[training](../usage.md#training) and [QAT](../usage.md#quantization-aware-training-qat)
commands.

### Training loss

For NSS v1, set `train` → `loss_fn` to `loss_v1`.

`train` → `loss_args` contains optional overrides. Omitting the dictionary, or
omitting an individual key, uses the built-in NSS v1 defaults. The generated
NSS configuration includes the recommended starting settings. Unknown or
misspelled keys are ignored, so check key names carefully.

### Dataset appearance

- `dataset` → `exposure`: Logarithmic exposure override, to brighten training images before tonemapping takes place. For example, `2` multiplies brightness by `e²`. A null or absent value takes exposure values from the dataset.

- `dataset` → `tonemapper`: Selects the tonemapping method:

  - `reinhard`: Simple, robust HDR compression that reduces each color channel independently; a good default when you want predictable brightness control.
  - `karis`: Compresses brightness using the strongest RGB channel. Tends to preserve color balance better in bright highlights.
  - `log`: Applies natural-log compression, strongly reducing bright values while keeping darker detail relatively visible.
  - `log10`: Applies base-10 log compression. Similar to `log` but produces lower brightness values.
  - `log_norm`: Applies log compression normalized to the expected HDR range, useful when you want values mapped into a more consistent bounded scale.
  - `aces`: Uses a filmic ACES-style curve with smooth highlight roll-off, useful when perceptual image appearance matters more than simple numeric compression.
