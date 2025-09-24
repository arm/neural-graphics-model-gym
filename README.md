<!---
SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->
# Neural Graphics Model Gym

> [!NOTE]
> Please be aware that this is a beta release. Beta means that the product may not be functionally or feature complete. At this early phase the product is not yet expected to fully meet the quality, testing or performance requirements of a full release. These aspects will evolve and improve over time, up to and beyond the full release. We welcome your feedback.

## Table of contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
    * [Setup locally](#setup-locally)
    * [Docker® image](#how-to-build-a-docker-image-for-neural-graphics-model-gym)
    * [(Optional) Sample datasets and pretrained weights](#optional-sample-datasets-and-pretrained-weights)
4. [Usage](#usage)
    * [CLI](#command-line)
    * [Python® package](#python-package)
    * [Profile training](#profile-training)
    * [TensorBoard](#tensorboard)
    * [Logging](#logging)
5. [Testing](#testing)
6. [Generating new training data](#generating-new-training-data)
7. [Code contributions](#code-contributions)
8. [Troubleshooting](#troubleshooting)
9. [Security](#security)
10. [License](#license)
11. [Trademarks and copyrights](#trademarks-and-copyrights)

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

### Setup locally

We recommend doing development within a **virtual environment** (e.g. using venv).

#### 1. Create and activate a Python venv:

```bash
python -m venv venv
source venv/bin/activate
```

#### 2. Available local installation options:

**Standard installation** -
Installs the package into your active environment:

```bash
make install
```

**Editable installation** -
Enables live package edits and includes dev tools and dependencies:

```bash
make install-dev
```

**Wheel Installation** -
Build and install a Python wheel:

```bash
make build-wheel
pip install dist/ng_model_gym-{version}-py3-none-any.whl
```

### How to build a Docker image for Neural Graphics Model Gym

A Dockerfile to help build and run Neural Graphics Model Gym is [provided](./Dockerfile).

This will build and install everything required, such as Python and system packages, into a Docker image.

To create a Docker image for Neural Graphics Model Gym, including all the required dependencies, a shell script is provided.

Run the following command to build the Docker image:

```bash
bash build_docker_image.sh
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

To quickly try out the repository without preparing your own data, run:

```bash
# Install huggingface_hub (if not already installed)
make install-dev

# Download pretrained model weights and datasets from Hugging Face
make test-download
```

* Example datasets will be placed under `tests/usecases/nss/datasets/`.
* Pretrained weights will be placed under `tests/usecases/nss/weights/`.

### Windows (experimental)

Support for Windows is experimental. Known limitations:

* There is no pre-built ML SDK Model Converter binary for Windows, so exporting a VGF will require additional setup steps.

There are no prebuilt ExecuTorch wheels for Windows,
so it must be downloaded as source code and added to `PYTHONPATH`. Make sure to get the version of the source code that corresponds with the build version specified in the `pyproject.toml`.
As the C++/runtime part of ExecuTorch isn't needed (only the ahead-of-time Python part is needed)
there is no need to build the downloaded source. Note that `PYTHONPATH` needs to be set to the folder _above_ the `executorch` folder.

For example (in PowerShell):

```
cd <folder>
git clone --depth=1 https://github.com/pytorch/executorch.git --branch 737916343a0c96ec36d678643dc01356a995f388
cd executorch
git submodule update --init --recursive -q
$env:PYTHONPATH=<folder>
```

ExecuTorch also has some Python dependencies that we need to install. These can be found from ExecuTorch's pyproject.toml,
which the following PowerShell snippet will extract and install:

```
cd executorch
pip install @([regex]::Matches((Get-Content pyproject.toml -Raw), '(?s)dependencies\s*=\s*\[(.*?)\]').Groups[1].Value | % { [regex]::Matches($_, '"([^"]+)"') | % { $_.Groups[1].Value } })
```

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
trained_model, trained_model_path = ngmg.do_training(config, ngmg.TrainEvalMode.FP32)
ngmg.do_evaluate(config, trained_model_path, ngmg.TrainEvalMode.FP32)

# Export the trained fp32 model to a VGF file.
ngmg.do_export(config, trained_model_path, export_type=ngmg.ExportType.FP32)
```

Jupyter® notebook tutorials on how to use the package, including:
* Training
* Quantization-aware training and exporting
* Evaluation
* Fine-tuning

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

A collection of unit and integration tests are provided to ensure the functionality of Neural Graphics Model Gym. The tests depend on pretrained weights and datasets from Hugging Face. To automatically download the required files, run the following make commands:

```bash
# Install additional dependencies required for testing
make install-dev

# Download pretrained model weights and datasets from Hugging Face
make test-download
```

To run all tests:

```bash
# Run all tests
make test

# Run unit tests
# (Test individual functions)
make test-unit

# Run integration tests
# (Test how parts of the application work together)
make test-integration
```

To run unit tests for a specific use case (e.g. NSS):

```bash
make test-unit USECASE=nss
```

To run unit tests from one specific file with tests:

```bash
python -m unittest tests.core.unit.utils.test_checkpoint_utils
```

To run integration tests for a specific use case:

```bash
make test-integration USECASE=nss
```

To run export integration tests:
```bash
make test-export
```

To see the test coverage report:

```bash
make coverage
```

To see all available commands:

```bash
make list
```

## Generating new training data

To train the Neural Super Sampling model, you will first need to capture training data from your game engine in the format expected by the model.

Documentation is provided [here](./docs/nss_dataset_specification.md) that goes into detail for the expected format and specification of the dataset.

Once you have captured data from your game engine, and it is in the expected format, then you can use the provided script [here](./scripts/safetensors_generator/safetensors_writer.py) to convert captured EXR frames to Safetensors.

This script can be run using the following:

```bash
python -m scripts.safetensors_generator.safetensors_writer -src="path/to/exr/root/dir"
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

Please see the documentation [here](./docs/nss_dataset_specification.md) for more details on the expected dataset format and layout.

## Code contributions
The full contribution guide can be seen in [CONTRIBUTING.md](./CONTRIBUTING.md).

Before making a pull request for any code changes, you must run the following checks:

```bash
make test       # Run all tests
make format     # Format files
make lint       # Lints src files
make lint-test  # Lints test files
make coverage   # Create coverage report
make bandit     # Run security check
```

### pre-commit module

pre-commit is used to run the checks mentioned above when making a new commit.

pre-commit will be installed when running

```bash
make install-dev
```

or it can be manually installed with

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

### Order of Outputs changes after exporting to a VGF or TOSA file
Currently when exporting to a VGF file you may see that the order of outputs is different to those specified in your PyTorch module. This is a bug and a fix will be made very soon to our project dependencies.

## Security

Arm takes security issues seriously, please see [SECURITY.md](SECURITY.md) for more details.

You can run the checker for security vulnerabilities with the following command:

```bash
make bandit
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