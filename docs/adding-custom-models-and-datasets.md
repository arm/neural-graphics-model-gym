<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->
# Adding custom models and datasets

Neural Graphics Model Gym supports adding your own custom models and datasets, enabling you to use them across all workflows.

When using the [CLI](../README.md#command-line-usage), new models and datasets should be added within the [usecases](../src/ng_model_gym/usecases/) directory. Each new subdirectory must contain an `__init__.py` file, and the model or dataset must be marked with a decorator, as described below, to be discovered.

If using the Neural Graphics Model Gym [Python package](../README.md#usage-as-a-python-package), new models and datasets can be placed anywhere in your project. They must also be marked with a decorator and the files containing them should be imported into your code in order for them to be registered.

#### Registering a custom model

To add a new model, mark the model class with the `@register_model()` decorator, giving the name and optional version to register it under. Models must inherit from the [`BaseNGModel`](../src/ng_model_gym/core/model/base_ng_model.py) class, implement any required methods, and their constructors must accept `params` as an argument.
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

To use a custom model or dataset, the configuration file must be updated with the model or dataset name and optional version it was registered with such as:

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

# Adding custom use cases

Neural Graphics Model Gym supports defining new custom use cases to group related models, datasets, configurations, and any additional required code together.

#### Where to add new use cases
To create a new use case when using the [CLI](../README.md#command-line-usage), add a folder under the existing [usecases](../src/ng_model_gym/usecases/) directory containing an `__init__.py` file.

When using the [Python package](../README.md#usage-as-a-python-package), new use cases can be added anywhere in your project as long as the model and dataset are imported into your code.

The required model implementation and dataset must be added, following [Adding custom models and datasets](#adding-custom-models-and-datasets) above, along with any additional pipeline code. The [nss](../src/ng_model_gym/usecases/nss) use case folder can be used as an example.

The use case logic is executed when its model and dataset are used in the configuration file being provided.

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

The [core](../src/ng_model_gym/core/) directory contains all code shared across use cases, including base classes, utilities, trainers, evaluators, loss functions, learning rate schedulers, and optimizers.
