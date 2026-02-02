<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->

# Setup
1. [Local setup](#local-setup)
    * [Linux](#linux)
    * [Windows (experimental)](#windows-experimental)
2. [Using Python wheel](#wheel-installation)
3. [Docker](#docker)
4. [Downloading sample data](#downloading-sample-datasets-and-pretrained-weights-optional)

## Local setup
### Linux
First, clone the repo:
```bash
git clone https://github.com/arm/neural-graphics-model-gym.git
```
Then create and activate a Python **virtual environment** (e.g. using venv).

Next, install the package into your active environment.

```bash
pip install .
```
A development installation can also be installed. This enables live package edits and includes dev tools and dependencies for uses such as testing, linting and code checking.
#### Dev installation
<details>
<summary>Editable dev installation</summary>

 An editable installation for development can be created using [Hatch](https://github.com/pypa/hatch).

First install Hatch, either following the installation steps [here](https://github.com/pypa/hatch/blob/master/docs/install.md), or by running: <!-- #blocklint: pragma -->

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
</details>

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

Then follow the setup steps in [Local setup](#local-setup).

## **Wheel Installation**
To build and install a Python wheel of the project using Hatch:

First follow the instructions to create an [editable installation](#dev-installation), then run:

```bash
hatch build-wheel
pip install dist/ng_model_gym-{version}-py3-none-any.whl
```

## **Docker**
### Build the Docker image

A Dockerfile to help build and run Neural Graphics Model Gym is [provided](../Dockerfile). This will build and install everything required, such as Python and system packages, into a Docker image.

Run the following command to build the Docker image:

```bash
docker build . -t neural-graphics-model-gym:latest -f Dockerfile
```

### Run the Docker container

To run the container, specify the shared memory size and access to all GPUs using the following command:
```bash
docker run -it --shm-size=2gb --gpus all neural-graphics-model-gym:latest
```

You can then run any of the commands in the next sections from within your Docker container.

#### Increase shared memory
To run training with the Docker image, the shared memory size must be increased from the default allocated to a container by using the `--shm-size` flag.

#### Access GPU in Docker container
To access your GPU inside the Docker container, follow instructions for your specific GPU.

For NVIDIA GPUs, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) then follow the instructions for [Configuring Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker). The GPU can then be accessed from the Docker container using the `--gpus` flag, either specifying `--gpus all` for access to all GPUs or for specific GPUs pass the device numbers in a comma separated list e.g. `--gpus "device=0,2"`.


## Downloading sample datasets and pretrained weights (Optional)
Sample datasets and pretrained weights are provided on Hugging Face, mainly used for [testing](./testing.md) and demonstration purposes.

To quickly try out the repository without preparing your own data, first follow the [dev installation](#dev-installation) section, then run:

```bash
# Download pretrained model weights and datasets from Hugging Face
hatch run test-download
```

* Example datasets will be placed under `tests/usecases/nss/datasets/`.
* Pretrained weights will be placed under `tests/usecases/nss/weights/`.