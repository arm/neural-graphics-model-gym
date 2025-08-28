<!---
SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->

# Contribution Guidelines

The Neural Graphics Model Gym project is open for external contributors and welcomes contributions.
Neural Graphics Model Gym is licensed under the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license and all accepted contributions must have the same license.

Below is an overview on contributing code to Neural Graphics Model Gym.

## Contributing code to Neural Graphics Model Gym

- Before the Neural Graphics Model Gym project accepts your contribution, you need to certify its origin and give us your permission. To manage this process we use
  [Developer Certificate of Origin (DCO) V1.1](https://developercertificate.org/).
  To indicate that contributors agree to the terms of the DCO, it's necessary to "sign off" the
  contribution by adding a line with name and e-mail address to every git commit message:

  ```log
  Signed-off-by: FIRST_NAME SECOND_NAME <your@email.address>
  ```

  This can be done automatically by adding the `-s` option to your `git commit` command.
  You must use your real name, no pseudonyms or anonymous contributions are accepted

- In each source file, include the following copyright notice:

  ```bash
  # SPDX-FileCopyrightText: Copyright <years changes were made> <copyright holder>.
  # SPDX-License-Identifier: Apache-2.0
  ```
  Note: if an existing file does not conform, please update the license header as part of your contribution.

## Code Reviews

Contributions must go through code review on GitHub.

Only reviewed contributions can go to the main branch of Neural Graphics Model Gym.

## Continuous integration

Contributions to Neural Graphics Model Gym go through testing on the Arm CI system. All unit, integration and regression tests must pass before a contribution gets merged to the Neural Graphics Model Gym main branch.
