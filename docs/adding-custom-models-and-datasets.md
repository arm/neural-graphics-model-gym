<!---
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
--->
# Adding custom models and datasets

Neural Graphics Model Gym supports adding your own custom models and datasets.

There are two common ways to add models and datasets:

1. Modify a prebuilt model (for example, NSS) by copying its implementation, registering it under a new name, and using its model config file as a starting point.
2. Create a new custom model and dataset from scratch, register, and use a custom config file.


#### Modifying a prebuilt model

If you copy a prebuilt model (such as NSS), modify the class decorator to register it under a new name (for example, `MyNSSModel`). Use the prebuilt model config, set `model.name/model.version` to your new registration, and keep `model_source: "prebuilt"`.

```bash
ng-model-gym init nss [save_dir]
```

```json
{
  "model": {
    "name": "nss",
    "model_source": "prebuilt",
    "version": "1",
    "scale": 2.0,
    "recurrent_samples": 16
  },
  "dataset": {
    "name": "NSS",
    "version": "1"
  }
}
```

If you are adding a new model or dataset from scratch, use the custom template as described in a [later section](#creating-custom-configuration-files).

#### Registering a new model

To add a new model, mark the model class with the `@register_model()` decorator, giving the name and optional version to register it under. Models must inherit from the [`BaseNGModel`](../src/ng_model_gym/core/model/base_ng_model.py) class, implement any required methods, and their constructors must accept `params` as an argument.

```python
from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.model.model_registry import register_model
from ng_model_gym.core.config.config_model import ConfigModel


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

#### Model and dataset discovery

Depending on how you interact with ng-model-gym (CLI or API), model and dataset discovery works differently. Registration happens when the module containing the decorated class is imported.

##### Using the CLI

Models/datasets registered under the `ng_model_gym/usecases` folder are auto-discovered. Outside of this folder, the CLI does not search for registered models/datasets. 

A suggested layout is:

```
new_usecase/
  ├── configs/
  │   └── my_model_template.json
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


##### Using the ng-model-gym API

If you interact with the model-gym using the API, registered models/datasets can live anywhere, but you must import their modules in your own code.

#### Creating custom configuration files

To create a configuration file for a custom model, use the CLI `init custom` command:

```bash
ng-model-gym init custom [save_dir]
```
Custom model configs must set `"model_source": "custom"`. This allows for extra JSON fields under the `model` section for use in your registered model code. Update the config with the names of your registered model and dataset.

```json
{
  "model": {
    "name": "registered_model_name",
    "model_source": "custom",
    "version": "1",
    "my_field": "value",
    "my_field2": "value"
  },
  "dataset": {
    "name": "dataset_name",
    "version": "1",
    ...
  }
}
```
