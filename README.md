<!---
SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->
<h1 align="center">Neural Graphics Model Gym</h1>
<p align="center">
<a href="https://huggingface.co/Arm/neural-super-sampling"><img alt="Model Card" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-neural%20super%20sampling%20-blue"></a>
<a href="https://github.com/arm/neural-graphics-model-gym/blob/main/LICENSE.md"><img alt="License" src="https://img.shields.io/badge/license-Apache%20License%202.0-green"></a>
<img alt="Python versions" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue">
<img alt="Neural Graphics Model Gym CLI help output" src="docs/ng-model-gym-cli-hero-img.png" width="665" height="284">
</p>

> [!NOTE]
> Please be aware that this is a beta release. Beta means that the product may not be functionally or feature complete. At this early phase the product is not yet expected to fully meet the quality, testing or performance requirements of a full release. These aspects will evolve and improve over time, up to and beyond the full release. We welcome your feedback.

## Table of contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
    * [Windows (experimental)](#windows-experimental)
    * [Setup locally](#setup-locally)
      * [Standard installation](#standard-installation)
      * [Editable installation with Hatch](#editable-installation-with-hatch)
      * [Wheel installation](#wheel-installation)
    * [Docker® image](#how-to-build-a-docker-image-for-neural-graphics-model-gym)
    * [(Optional) Sample datasets and pretrained weights](#optional-sample-datasets-and-pretrained-weights)
4. [Usage](#usage)
    * [CLI](#command-line)
    * [Python® package](#python-package)
    * [Profile training](#profile-training)
    * [TensorBoard](#tensorboard)
    * [Logging](#logging)
5. [Testing](#testing)
6. [Adding custom models and datasets](#adding-custom-models-and-datasets)
7. [Adding custom use cases](#adding-custom-use-cases)
8. [Generating new training data](#generating-new-training-data)
9. [Code contributions](#code-contributions)
10. [Troubleshooting](#troubleshooting)
11. [Security](#security)
12. [License](#license)
13. [Trademarks and copyrights](#trademarks-and-copyrights)

## Introduction

**Neural Graphics Model Gym**  is a Python® toolkit for developing real-time Neural Graphics machine learning models.

With **Neural Graphics Model Gym** you can train, finetune and evaluate your Neural Graphics models.
**Neural Graphics Model Gym** also enables you to perform quantization of your model before exporting it to a format compatible with ML extensions for Vulkan® - allowing you to run on the latest mobile devices.

Currently, we include the following Neural Graphics use cases:

* Neural Super Sampling (NSS)
  * NSS allows for high-fidelity, real-time graphics in game engines. By feeding low-resolution frames, along with spatial and motion information, into a neural network we are able to construct high-resolution frames that suffer no loss in quality.

## Prerequisites

To build and run Neural Graphics Model Gym, the following are required:

* Ubuntu® >= 22.04
  * Neural Graphics Model Gym has been tested on 22.04 LTS and 24.04 LTS, but should work on other Linux® distributions
* 3.10 <= Python < 3.13
* Python development package (e.g. `python3-dev`)
* NVIDIA® CUDA® capable GPU
* CUDA Toolkit v12.8 or later
* Git LFS

## Setup

### Windows (experimental)

Support for Windows is experimental. Known limitations:

* To be able to use `torch.compile` on Windows, a C++ compiler is required.

#### Install Visual Studio 2022:
1. [Install Visual Studio 2022](https://visualstudio.microsoft.com/vs/older-downloads/), making sure **Workloads > Desktop & Mobile > Desktop Development with C++** is checked during installation.

2. Launch `Visual Studio 2022 Developer Command Prompt` or `x64 Native Tools Command Prompt for VS 2022`.

3. Verify the compiler is found:
    ```bat
    cl.exe
    ```

Follow the Windows specific setup steps in [Setup locally](#setup-locally).

### Setup locally

#### **Standard installation**

We recommend doing development within a **virtual environment** (e.g. using venv).

**1. Create and activate a Python venv:**

Linux:
```bash
# Create a virtual environment
python -m venv venv

# Activate it
source venv/bin/activate
```

Windows:
```bash
# Create a virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

**2. Install the package into your active environment:**

```bash
pip install .
```

#### **Editable dev installation with Hatch**
An editable installation for development can be created using [Hatch](https://github.com/pypa/hatch).

The development environment enables live package edits and includes dev tools and dependencies for uses such as testing, linting and code checking.

First install Hatch, either following the installation steps [here](https://github.com/pypa/hatch/blob/master/docs/install.md), or by running:

```bash
pip install hatch==1.16.2
```

Then create the Hatch environment with:

```bash
hatch -v env create
```

Enter the Hatch development environment with:
```bash
hatch shell
```

To exit the Hatch environment, run:
```bash
exit
```

To remove the Hatch environment, run:
```bash
hatch env remove
```

#### **Wheel Installation**
Build and install a Python wheel using Hatch:

First follow the [editable installation with Hatch](#editable-installation-with-hatch) section above, then run:

```bash
hatch build-wheel
pip install dist/ng_model_gym-{version}-py3-none-any.whl
```

### How to build a Docker image for Neural Graphics Model Gym

A Dockerfile to help build and run Neural Graphics Model Gym is [provided](./Dockerfile).

This will build and install everything required, such as Python and system packages, into a Docker image.

To create a Docker image for Neural Graphics Model Gym, including all the required dependencies, a shell script is provided.

Run the following command to build the Docker image:

```bash
docker build . -t ng-model-gym-image -f Dockerfile
```

#### Increase shared memory
To run training with the Docker image, the shared memory size must be increased from the default allocated to a container by using the `--shm-size` flag.

#### Access GPU in Docker container
To access your GPU inside the Docker container, follow instructions for your specific GPU.

For NVIDIA GPUs, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) then follow the instructions for [Configuring Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker). The GPU can then be accessed from the Docker container using the `--gpus` flag, either specifying `--gpus all` for access to all GPUs or for specific GPUs pass the device numbers in a comma separated list e.g. `--gpus "device=0,2"`.

#### Run the Docker container

To run the container, specify the shared memory size and access to all GPUs using the following command:
```bash
docker run -it --shm-size=2gb --gpus all ng-model-gym-image
```

You can then run any of the commands in the next sections from within your Docker container.

### (Optional) Sample datasets and pretrained weights
Sample datasets and pretrained weights are provided on Hugging Face, mainly used for [testing](#testing) and demonstration purposes.

To quickly try out the repository without preparing your own data, first follow the [editable installation with Hatch](#editable-installation-with-hatch) section, then run:

```bash
# Download pretrained model weights and datasets from Hugging Face
hatch test-download
```

* Example datasets will be placed under `tests/usecases/nss/datasets/`.
* Pretrained weights will be placed under `tests/usecases/nss/weights/`.


## Usage

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

To load a set of previously trained model weights and perform finetuning:

Specify the path to the pretrained weights file in your [configuration file](#configuration-file) under

```json
"train": {
  "pretrained_weights": "path/to/pretrained_weights.pt"
}
```

Then run:

```bash
ng-model-gym -c <path/to/config/file> train --finetune
```

To resume training from the latest saved checkpoint specified in your configuration file, run:

```bash
ng-model-gym -c <path/to/config/file> train --resume
```

Other actions can be specified using additional flags.

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

### Profile training

#### Trace profiler
For detailed performance metrics, you can enable the trace profiler.

```bash
# PyTorch profiling
ng-model-gym --profiler=trace -c <path/to/config/file> train
```
This produces a JSON file viewable in a browser at https://ui.perfetto.dev/

#### GPU memory profiler

Measure GPU memory usage during training.

```bash
# Profiling CUDA memory usage.
ng-model-gym --profiler=gpu_memory -c <path/to/config/file> train
```

The output pickle file is viewable in a browser at https://pytorch.org/memory_viz

### TensorBoard
[TensorBoard](https://www.tensorflow.org/tensorboard) provides visualization and tooling needed for machine learning experimentation. Use the following command to see logs with TensorBoard once you have started training:

```bash
tensorboard --logdir=tensorboard-logs
```

By default, the training logs will be written to the `tensorboard-logs` directory in the current working directory. This can be changed in your configuration file.

### Logging

By default, logging is enabled and set to INFO mode, which will print helpful information during execution.
All logs will be written to an `output.log` file located within the output directory specified in the configuration file.
The logging mode is customizable by using flags with the `ng-model-gym` CLI command. See the options below for examples.

`--log-level=quiet` can be added to silence all logs, except errors.

```bash
ng-model-gym --log-level=quiet -c <path/to/config/file> train
```

`--log-level=debug` can be added to print even more information during the process.

```bash
ng-model-gym --log-level=debug -c <path/to/config/file> train
```

Logging can also be specified when importing the package as follows.

```python
import ng_model_gym as ngmg
from pathlib import Path

# Create a Config object using path to a configuration file
parameters = ngmg.load_config_file(Path("/path/to/config"))

# Enable logging for ng_model_gym
ngmg.logging_config(parameters)
```

## Testing

A collection of unit and integration tests are provided to ensure the functionality of Neural Graphics Model Gym.

Testing can be run using Hatch commands.
First [install Hatch and create an environment](#editable-installation-with-hatch).

This will install all the dependencies for Neural Graphics Model Gym, plus the additional dependencies required for testing.

The tests depend on pretrained weights and datasets from Hugging Face. To automatically download the required files, run the following command:

```bash
hatch run test-download
```

**To run tests:**

```bash
# Run all tests
hatch run test:test

# Run all unit tests
# (Test individual functions)
hatch run test:unit

# Run all integration tests
# (Test how parts of the application work together)
hatch run test:integration

# Run export tests
hatch run test:export
```

To run tests across all supported Python versions, you can run commands like the following:
```bash
# Run all tests across all supported Python versions
hatch run test-matrix:test

# Run unit tests across all supported Python versions
hatch run test-matrix:unit
```

**To run unit or integration tests for only a specific use case (e.g. NSS):**

First set an environment variable for the usecase, then run the test command:

Linux (bash):
```bash
export USECASE=nss && hatch run test:unit
```

Windows (cmd):
```bat
set USECASE=nss && hatch run test:unit
```

**To run unit tests from one specific file with tests:**

```bash
hatch run python -m pytest tests/core/unit/utils/test_checkpoint_utils.py
```

**To create a test coverage report:**

```bash
hatch run test:coverage-check
```

**All available commands in the Hatch environment are listed under scripts when running:**

```bash
hatch env show
```

## Adding custom models and datasets

Neural Graphics Model Gym supports adding your own custom models and datasets, enabling you to use them across all workflows.

When using the [CLI](#command-line), new models and datasets should be added within the [usecases](./src/ng_model_gym/usecases/) directory. Each new subdirectory must contain an `__init__.py` file, and the model or dataset must be marked with a decorator, as described below, to be discovered.

If using the Neural Graphics Model Gym [Python package](#python-package), new models and datasets can be placed anywhere in your project. They must also be marked with a decorator and the files containing them should be imported into your code in order for them to be registered.

#### Registering a custom model

To add a new model, mark the model class with the `@register_model()` decorator, giving the name and optional version to register it under. Models must inherit from the [`BaseNGModel`](./src/ng_model_gym/core/model/base_ng_model.py) class, implement any required methods, and their constructors must accept `params` as an argument.
```python
from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.model.model_registry import register_model
from ng_model_gym.core.utils.config_model import ConfigModel

@register_model(name="name", version="version")
class NewModel(BaseNGModel):
  def __init__(self, params: ConfigModel):
    super().__init__(params)

  ...
```

#### Registering a custom dataset

To add a new dataset, mark the dataset class with the `@register_dataset()` decorator, giving the name and optional version to register it under. Datasets must inherit from the `torch.utils.data.Dataset` class.
```python
from torch.utils.data import Dataset
from ng_model_gym.core.data.dataset_registry import register_dataset

@register_dataset(name="name", version="version")
class NewDataset(Dataset):
  ...
```

#### Updating the config to use a custom model or dataset

To use a custom model or dataset, the [configuration file](#configuration-file) must be updated with the model or dataset name and optional version it was registered with such as:

```json
{
  "model": {
    "name": "model_name",
    "version": "1"
  },
  "dataset": {
    "name": "dataset_name",
    "version": "1",
    ...
  }
}
```

## Adding custom use cases

Neural Graphics Model Gym supports defining new custom use cases to group related models, datasets, configurations, and any additional required code together.

#### Where to add new use cases
To create a new use case when using the [CLI](#command-line), add a folder under the existing [usecases](./src/ng_model_gym/usecases/) directory containing an `__init__.py` file.

When using the [Python package](#python-package), new use cases can be added anywhere in your project as long as the model and dataset are imported into your code.

The required model implementation and dataset must be added, following [Adding custom models and datasets](#adding-custom-models-and-datasets) above, along with any additional pipeline code. The [nss](./src/ng_model_gym/usecases/nss) use case folder can be used as an example.

The use case logic is executed when its model and dataset are used in the [configuration file](#configuration-file) being provided.

A suggested layout is:

```
new_usecase/
  ├── configs/
  │   └── config.json
  ├── data/
  │   ├── __init__.py
  │   └── dataset.py
  ├── model/
  │   ├── __init__.py
  │   ├── model.py
  │   └── ...
  └── __init__.py
```

The [core](./src/ng_model_gym/core/) directory contains all code shared across use cases, including base classes, utilities, trainers, evaluators, loss functions, learning rate schedulers, and optimizers.

## Generating new training data

To train the Neural Super Sampling model, you will first need to capture training data from your game engine in the format expected by the model.

The [data capture guide](./docs/nss/nss_data_capture_guide.md) explains how to capture data from your game and how to convert it into the expected format. This guide also contains recommendations regarding dataset size and types of game data that shpuld be captured.

Documentation is provided [here](./docs/nss/nss_dataset_specification.md) that goes into detail for the expected format and specification of the dataset.

Once you have captured data from your game engine, and it is in the expected format, then you can use the provided script [here](./scripts/safetensors_generator/safetensors_writer.py) to convert captured EXR frames to Safetensors.

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

Please see the documentation [here](./docs/nss/nss_dataset_specification.md) for more details on the expected dataset format and layout.

## Code contributions
The full contribution guide can be seen in [CONTRIBUTING.md](./CONTRIBUTING.md).

Before making a pull request for any code changes, you must run the following checks:

```bash
hatch run test            # Run all tests
hatch run format          # Format files
hatch run lint-all        # Lints src and test files
hatch run coverage-check  # Create coverage report
hatch run bandit-check    # Run security check
hatch run blocklint       # Run non-inclusive language check
hatch run copyright       # Run copyright header check
```

### pre-commit module

pre-commit is used to run the checks mentioned above when making a new commit.

pre-commit will be installed when following the [editable installation with Hatch](#editable-installation-with-hatch), or it can be manually installed with:

```bash
pip install pre-commit
```

To install the pre-commit git hook, run the following command

```bash
pre-commit install
```

To check that all the pre-commit checks run successfully, run the following command

```bash
pre-commit run --all-files
```

## Troubleshooting

### Out of Memory Errors
Occasionally, your machine might run out of GPU memory during model training or while running tests. To fix this, you will need to change some hyperparameters in your configuration JSON. For example:
* **Lower the batch size** by using a smaller value for `train.batch_size`. This reduces the number of samples being processed at once.
* **Use fewer recurrent samples** by decreasing the value of `dataset.recurrent_samples`.
* **Reduce the number of workers** by reducing `dataset.num_workers`. This will increase the time it takes to perform training/evaluation/etc. but the machine is less likely to run out of memory since there are fewer tasks being run in parallel.

### Slow model training in WSL2 or Windows
A common cause for training running slower than expected in WSL2 or Windows is the GPU running out of device memory and falling back to slower shared system memory. To address this, reduce memory usage as described in the **Out of Memory Errors** section.

## Security

Arm takes security issues seriously, please see [SECURITY.md](SECURITY.md) for more details.

After following [editable installation with Hatch](#editable-installation-with-hatch), you can run the checker for security vulnerabilities with the following command:
```bash
hatch run bandit-check
```

## License

Neural Graphics Model Gym is licensed under [Apache License 2.0](LICENSE.md).

## Trademarks and copyrights

* Linux® is the registered trademark of Linus Torvalds in the U.S. and elsewhere.
* Python® is a registered trademark of the PSF.
* Ubuntu® is a registered trademark of Canonical.
* Docker and the Docker logo are trademarks or registered trademarks of Docker, Inc. in the United States and/or other countries. Docker, Inc. and other parties may also have trademark rights in other terms used herein.
* NVIDIA and the NVIDIA logo are trademarks and/or registered trademarks of NVIDIA Corporation in the U.S. and other countries.
* “Jupyter” and the Jupyter logos are trademarks or registered trademarks of LF Charities.
* Vulkan is a registered trademark and the Vulkan SC logo is a trademark of the Khronos Group Inc.
* Microsoft, Windows are trademarks of the Microsoft group of companies