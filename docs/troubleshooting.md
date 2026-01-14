<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->

# Troubleshooting
Known issues with workarounds:
* [Out of Memory Errors](#out-of-memory-errors)
* [Slow model training in WSL2 or Windows](#slow-model-training-in-wsl2-or-windows)
## Out of Memory Errors
Occasionally, your machine might run out of GPU memory during model training or while running tests. To fix this, you will need to change some hyperparameters in your configuration JSON. For example:
* **Lower the batch size** by using a smaller value for `train.batch_size`. This reduces the number of samples being processed at once.
* **Use fewer recurrent samples** by decreasing the value of `dataset.recurrent_samples`.
* **Reduce the number of workers** by reducing `dataset.num_workers`. This will increase the time it takes to perform training/evaluation/etc. but the machine is less likely to run out of memory since there are fewer tasks being run in parallel.

## Slow model training in WSL2 or Windows
A common cause for training running slower than expected in WSL2 or Windows is the GPU running out of device memory and falling back to slower shared system memory. To address this, reduce memory usage as described in the **Out of Memory Errors** section.