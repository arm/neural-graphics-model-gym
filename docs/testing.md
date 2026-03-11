<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->

# Testing

**To download pretrained weights and datasets:**

The tests depend on pretrained weights and datasets from Hugging Face. To automatically download the required files, run the following command:
```bash
hatch run test:download
```

**To run tests:**

```bash
# Run all tests
hatch run test:test

# Run all unit tests
# (Test individual functions)
hatch run test:unit

# Run core unit tests only
hatch run test:unit-core

# Run usecase unit tests only
hatch run test:unit-usecase-all

# Run all integration tests
# (Test how parts of the application work together)
hatch run test:integration

# Run core integration tests only
hatch run test:integration-core

# Run usecase integration tests only
hatch run test:integration-usecase-all

# Run export tests
hatch run test:export
```

**To run tests across all supported Python versions:**
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
export USECASE=nss && hatch run test:unit-usecase
export USECASE=nss && hatch run test:integration-usecase
```

Windows (cmd):
```bat
set USECASE=nss && hatch run test:unit-usecase
set USECASE=nss && hatch run test:integration-usecase
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