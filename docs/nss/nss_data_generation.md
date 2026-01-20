<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->

# Generating new training data

To train the Neural Super Sampling model, you will first need to capture training data from your game engine in the format expected by the model.

The [data capture guide](./nss_data_capture_guide.md) explains how to capture data from your game and how to convert it into the expected format. This guide also contains recommendations regarding dataset size and types of game data that shpuld be captured.

Documentation is provided [here](./nss_dataset_specification.md) that goes into detail for the expected format and specification of the dataset.

Once you have captured data from your game engine, and it is in the expected format, then you can use the provided script [here](../../scripts/safetensors_generator/safetensors_writer.py) to convert captured EXR frames to Safetensors.

This script can be run using the following:

```bash
python -m scripts.safetensors_generator.safetensors_writer -src="path/to/exr/root/dir" -reader=EXRv101 -extension=exr
```

These Safetensors files can then be cropped by running the script passing in the directory containing the uncropped Safetensors as the source:

```bash
python -m scripts.safetensors_generator.safetensors_writer -src="path/to/safetensors/root/dir" -reader=cropper -extension=safetensors
```

Additional optional flags:

| Flag        | Description | Default |
|-------------| ------- |---------|
| `-dst`      |Path to root folder of destination| `./output/safetensors` |
| `-threads`  |Number of parallel threads to use | `1`     |
| `-extension`     |File extension of the source data  | `"exr"` |
| `-overwrite`     | Overwrite data in destination if it already exists| `False` |
| `-linear-truth` |  Whether the ground truth is already linear; assumes Karis TM if not| `True`  |
| `-logging_output_dir` | Path to folder for logging output| `./output` |
| `-reader` | Name of the data reader to use | `"EXRv101"` |
| `-crop_size` | Crop size in `outDims` | `256` |

Please see the documentation [here](./nss_dataset_specification.md) for more details on the expected dataset format and layout.
