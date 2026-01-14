<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->
# Usage

There are two ways to use Neural Graphics Model Gym after package installation:

1. [Command line tool](#command-line)
2. [Imported as a Python package](#python-package)

### Command-line

To view all available flags and commands, run:

```bash
ng-model-gym --help
```

#### Configuration file

Neural Graphics Model Gym is configured using a JSON file, which contains all necessary parameters such as hyperparameters and paths to datasets.

To generate a configuration file, run:
```bash
ng-model-gym init
```

This command creates two files in your working directory:

- `config.json`
    - Template populated with default values. Some entries have placeholder values (e.g. "<...>"). Make sure to replace those with your own settings. Dataset paths are expected to be folders containing datasets, not individual files.

- `schema_config.json`
  -  An accompanying file detailing all available configuration parameters

Use your custom configuration when invoking CLI commands by providing its path with the `--config-path` or `-c` flag as shown below:

```bash
ng-model-gym --config-path=<path/to/config.json> train
```

The `--config-path` (or `-c`) flag is **required** when running the `train`, `qat`, `evaluate`, or `export` commands.
These commands will fail if a valid config file path is not provided.

You can list all available configuration parameters in the CLI with the command below, but it is preferable to use `schema_config.json`.

```bash
ng-model-gym config-options
```

#### Training
> **The --config-path (or -c) flag is required when running this command.**

To perform training and evaluation, run:

```bash
ng-model-gym -c <path/to/config/file> train
```

To perform training without evaluation, run:

```bash
ng-model-gym -c <path/to/config/file> train --no-evaluate
```

To fine-tune from an existing checkpoint, pass the weights path directly:

```bash
ng-model-gym -c <path/to/config/file> train --finetune path/to/pretrained_weights.pt
```

To resume training from an existing checkpoint file or a directory containing checkpoints, run:
```bash
ng-model-gym -c <path/to/config/file> train --resume path/to/checkpoint.pt
ng-model-gym -c <path/to/config/file> train --resume path/to/checkpoint_dir/
```

`--finetune` and `--resume` flags are mutually exclusive and should not be set at the same time. Other actions can be specified using additional flags.

To see all available flags, run:

```bash
ng-model-gym train --help
```

#### Evaluation
> **The --config-path flag is required when running this command.**

To perform evaluation of a trained model, run:

```bash
ng-model-gym -c <path/to/config/file> evaluate --model-path=<path/to/model.pt> --model-type=<fp32|qat_int8>
```

Ensure you select the correct `--model-type` to match the format of your saved model.

#### Quantization aware training (QAT)
> **The --config-path flag is required when running this command.**

Neural Graphics Model Gym supports quantization aware training.

To perform QAT, run:

```bash
ng-model-gym -c <path/to/config/file> qat
```

To perform QAT without evaluation, run:

```bash
ng-model-gym -c <path/to/config/file> qat --no-evaluate
```

To load a set of previously trained model weights and perform finetuning, run:

```bash
ng-model-gym -c <path/to/config/file> qat --finetune
```

To resume QAT from the latest saved checkpoint specified in your configuration file, run:

```bash
ng-model-gym -c <path/to/config/file> qat --resume
```

Other actions can be specified using additional flags.

To see all available flags, run:

```bash
ng-model-gym qat --help
```

#### Export
> **The --config-path flag is required when running this command.**

Neural Graphics Model Gym uses ExecuTorch with the Arm backend to export models to a VGF file.

To export a trained model to a VGF file, run:

```bash
ng-model-gym -c <path/to/config/file> export --model-path=<path/to/model.pt> --export-type=<fp32|qat_int8|ptq_int8>
```

Ensure you select an export-type of fp32, qat_int8, or ptq_int8 with `--export-type`. Only QAT trained models can be exported to qat_int8.

The configuration file specifies the output directory for the generated VGF file.

### Python package

The second way to use Neural Graphics Model Gym is to import it as a Python package.

The following snippet shows how to use the package to generate a config, perform training, evaluation and exporting the model.

```python
import ng_model_gym as ngmg

# Generate config file in specified directory using the API or CLI
# Note: The config file must be filled in before use
ngmg.generate_config_file("/save/dir")
```

```python
import ng_model_gym as ngmg
from pathlib import Path

# Create a Config object using path to a configuration file
# and extract parameters from it.
config = ngmg.load_config_file(Path("/path/to/config/file"))

# Enable logging for ng_model_gym
ngmg.logging_config(config)

# Do training and evaluation.
trained_model_path = ngmg.do_training(config, ngmg.TrainEvalMode.FP32)
ngmg.do_evaluate(config, trained_model_path, ngmg.TrainEvalMode.FP32)

# Export the trained fp32 model to a VGF file.
ngmg.do_export(config, trained_model_path, export_type=ngmg.ExportType.FP32)
```

Jupyter® notebook tutorials on how to use the package, including:
* Training
* Quantization-aware training and exporting
* Evaluation
* Fine-tuning
* Adding a custom model

can be found in the [neural-graphics-model-gym-examples](https://github.com/arm/neural-graphics-model-gym-examples) repository.