# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from torch import nn


class BaseNGModel(nn.Module, ABC):
    """
    Base class for creating neural-graphics models.

    Subclasses should:
        * Expose the core neural network via the method `get_neural_network()`
        * Write the model `forward()` pass

    Example::

        import torch

        class ExampleNGModel(BaseNGModel):
            def __init__(self):
                super().__init__()
                self.neural_network: nn.Module = ExampleNeuralNetwork()

            def get_neural_network(self) -> nn.Module:
                return self.neural_network

            def forward(self, input_data):
                x = self.preprocessing(input_data)
                x = self.neural_network(x)
                x = self.postprocessing(x)
                return x
    """

    def __init__(self) -> None:
        """Initialise PyTorch nn.Module"""
        super().__init__()

    @abstractmethod
    def get_neural_network(self) -> nn.Module:
        """
        Get the model's neural network. It should be the same network every call.

        Returns:
            nn.Module: The neural network performing the core forward computation.
        """
        raise NotImplementedError
