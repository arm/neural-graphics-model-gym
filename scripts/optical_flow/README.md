<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->
# Optical Flow example

This folder contains example and implementation scripts for BlockMatch-based optical flow.

## Example usage

From the root directory run the example with explicit image paths:

```bash
python scripts/optical_flow/optical_flow_example.py \
  --img-tm1 path/to/frame_tm1.png \
  --img-t path/to/frame_t.png \
  --out path/to/optical_flow_output.npy
```

The script loads the two RGB frames, builds an all-zero motion-vector hint
field, runs `BlockMatchV32`, and saves the optical flow output to disk as a
NumPy `.npy` file at the path provided by `--out` (default:
`optical_flow_output.npy`).
