<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->
# Monitoring training
[TensorBoard](https://www.tensorflow.org/tensorboard) provides visualization and tooling needed for machine learning experimentation. Use the following command to see logs with TensorBoard once you have started training:

```bash
tensorboard --logdir=tensorboard-logs
```

By default, the training logs will be written to the `tensorboard-logs` directory in the current working directory. This can be changed in your configuration file.

# Profiling

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
