<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->

# Generating new training data

To train the Neural Frame Rate Upscaling model, you will first need to capture training data from your game engine in the format expected by the model.

Once you have captured data from your game engine, and it is in the expected format, then you can use the provided script [here](../../scripts/safetensors_generator/safetensors_writer.py) to convert captured EXR frames to Safetensors.

## NFRU EXR to Safetensors

This script can be run for NFRU using the following:

```bash
python -m scripts.safetensors_generator.safetensors_writer -src="path/to/exr/root/dir" -reader=NFRUv2_2 -extension=exr
```

## Common flags

| Flag        | Description | Default |
|-------------| ------- |---------|
| `-dst`      |Path to root folder of destination| `./output/safetensors` |
| `-threads`  |Number of parallel threads to use | `1`     |
| `-extension`     |File extension of the source data  | `"exr"` |
| `-overwrite`     | Overwrite data in destination if it already exists| `False` |
| `-logging_output_dir` | Path to folder for logging output| `./output` |
| `-reader` | Name of the data reader to use | `"NSSv1_0_1"` |
| `-crop_size` | Crop size in `outDims` (cropper only) | `256` |

## NFRU-specific flags

| Flag        | Description | Default |
|-------------| ------- |---------|
| `-reader=NFRUv2_2` | NFRU EXR dataset reader | N/A |

Please see the documentation [here](./nfru_dataset_specification.md) for NFRU dataset format details.
