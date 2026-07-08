<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->
# How to configure NFRU models

Before reading further, please read `docs/usage.md`. That file explains how to configure and use Model Gym.


## Introduction

Model Gym's NFRU model requires a configuration file, typically named `nfru_config.json`. This file is initially generated using the following command:

    ng-model-gym init nfru

This document is a guide to the most common changes you will need to make to `nfru_config.json`. For more details, refer to `schema_config.json`, which contains a detailed description of every configuration setting.


## Data you must provide

It's essential to provide the following information:

- `dataset` → `path` → `train`: path to a directory containing training data in the form of a collection of `.safetensors` files.
- `dataset` → `path` → `validation`: path to a directory containing validation data.
- `dataset` → `path` → `test`: path to a directory containing test data.


## Other settings you may to wish to change

### Dynamic mask

The “dynamic mask” marks pixels whose motion appear to come from scene/object movement rather than camera movement. The key settings controlling this functionality are:

- `model` → `dynamic_mask_is_runtime_accurate`: If `true`, the dynamic mask implementation used in training/evaluation will better resemble the deployed runtime implementation. This might make preprocessing/motion-vector handling slightly slower.

- `model` → `mv_similarity_threshold`: Controls the sensitivity used when constructing the dynamic mask. A lower similarity threshold will more readily identify pixels as containing scene/object motion independent of camera motion; a higher similarity threshold will mark fewer pixels in this way.

  Most users should leave `mv_similarity_threshold` as `null`. This causes Model Gym to use a default threshold, based upon the setting of `dynamic_mask_is_runtime_accurate`.

### Augmentation

`dataset` → `color_preprocessing` controls image processing performed during the augmentation process.

Different settings are provided for training, validation and test datasets, although the format used to describe the processing is identical in each case. The different datasets are described underneath:

- `dataset` → `color_preprocessing` → `train`
- `dataset` → `color_preprocessing` → `validation`
- `dataset` → `color_preprocessing` → `test`

#### Exposure

The first step in augmentation is to adjust image exposure. This is controlled by several keys:

- `dataset` → `color_preprocessing` → ⟨DATASET⟩ → `exposure`: Brightens or darkens dataset images. Three options are available:

  - `auto`: directs Model Gym to compute exposure from average image brightness. Requires a target brightness to be configured – this is given in `auto_exposure_key_value`, below.
  - A real number, representing a fixed, logarithmic exposure value. For example, `2` multiplies brightness by `e²`.
  - A range `[min, max]`: randomly resamples exposure within the provided range.

- `dataset` → `color_preprocessing` → ⟨DATASET⟩ → `auto_exposure_key_value`: Only used when exposure is set to `auto`. Specifies a target brightness value, typically in the range 0.0 to 1.0.

- `dataset` → `color_preprocessing` → ⟨DATASET⟩ → `auto_exposure_variance`: Allows exposure to be varied for different time indexes. This is an advanced topic and not further specified here.

#### Color pipeline

After exposure adjustment, color adjustments can optionally be applied. These are specified in key `dataset` → `color_preprocessing` → ⟨DATASET⟩ → `pipeline`. Choose between the following:

- `reinhard`: Applies Reinhard tonemapping, compressing bright linear HDR values into a display-like range.
- `contrast`: Increases or decreases contrast around mid-gray.
- `saturation`: Increases or decreases color intensity while preserving brightness.
- `temperature_tint`: Applies a simple color shift.

Adjustments are applied in the order they are given in a list. For example, the following applies `"reinhard"`, then `"contrast"`, in that order:

    "pipeline": ["reinhard", "contrast"]

#### Pipeline parameters

Some pipeline stages can optionally be passed parameters:

- `contrast` and `saturation` take a `strength` parameter, where 1.0 leaves the image unchanged, below 1.0 decreases the effect and above 1.0 increases the effect. The default strength is 1.2 (should no parameters be supplied).
- `temperature_tint` accepts parameters `temperature` and `tint`:
  - `temperature` changes the warmth of colors. A value above zero increases red and reduces blue. A value below zero increases blue and reduces red. The default is no adjustment.
  - `tint` alters the green channel. Below zero adds green; above zero reduces green. The default is no adjustment.

Pass parameters using an object of this form:

    { "<stage_name>": { "<parameter_name>": <value> … } }

For example, to increase contrast by 10% and reduce saturation by 10%, you would write:

    "pipeline": [
        { "contrast": { "strength": 1.1 } },
        { "saturation": { "strength": 0.9 } }
    ]

#### Random stage groups in pipelines

You can apply pipeline stages randomly. This is mainly useful for training.

To do this, put the stages into a nested list. For example:

    "pipeline": [
        ["reinhard"],
        ["contrast", "saturation", "none"]
    ]

This means:

1. Always apply `reinhard`.
2. Then randomly choose one of:
   - adjust contrast
   - adjust saturation
   - do nothing.

Omitted parameters to stages in random stage groups are themselves randomly determined. Notice the use of the `none` keyword for “do nothing”.
