# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import logging
from abc import ABC, abstractmethod

from torch import nn

from ng_model_gym.core.model.base_ng_model import BaseNGModel

logger = logging.getLogger(__name__)


class BaseNGModelWrapper(nn.Module, ABC):
    """
    Base class for creating classes that contains BaseNGModel

    Subclasses should:
        * Implement getter/setter methods for the stored BaseNGModel
        * Write the model `forward()` pass

    Example::

        class ExampleWrapper(BaseNGModelWrapper):
            def __init__(self, ng_model: BaseNGModel):
                self.ng_model = BaseNGModel

            def get_ng_model(self) -> BaseNGModel:
                return self.ng_model

            def set_ng_model(self, ng_model: BaseNGModel):
                self.ng_model = ng_model

            def forward(self, input_data):
                ...
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_ng_model(self) -> BaseNGModel:
        """
        Get the model's neural graphics model. It should be the same model every call.

        Returns:
            BaseNGModel: The wrapped neural graphics model
        """
        raise NotImplementedError

    @abstractmethod
    def set_ng_model(self, ng_model: BaseNGModel) -> None:
        """
        Set the wrapper's neural graphics model
        Args:
            ng_model (BaseNGModel): Neural graphics model to set.
        """
        raise NotImplementedError
