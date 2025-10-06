# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ng_model_gym.core.model.base_ng_model import BaseNGModel
from ng_model_gym.core.utils.tensor_types import TensorData


class FeedbackModel(BaseNGModel):
    """Wrapper around model for recurrent feedback"""

    def __init__(self, ng_model: BaseNGModel, recurrent_samples, device: torch.device):
        super().__init__()

        self.nss_model = ng_model
        self.recurrent_samples = recurrent_samples
        self.unpad = True
        self.device = device
        self.history_buffers = self.nss_model.init_history_buffers()

    def get_neural_network(self) -> nn.Module:
        """Return the core trainable neural network"""
        return self.nss_model.get_neural_network()

    def set_neural_network(self, neural_network: nn.Module) -> None:
        self.nss_model = neural_network

    def forward(self, x):
        """Run forward pass for the recurrent model.
        Input is in channel-first format (N, T, C, H, W).
        """
        # Get Input Data for t=0
        inputs = self._get_input_data_at_t(x, t=0)

        # Initialise History Buffers Prev
        inputs = self.set_buffers(inputs)

        # Run first inference and initialize output `dict`
        y_pred = self.nss_model(inputs)
        outputs = {}

        # Update history buffers and unpad model predictions.
        y_pred_unpadded = self.update_buffers(inputs, y_pred)

        # Turn outputs into a dict containing lists of tensors.
        for key, value in y_pred_unpadded.items():
            outputs[key] = [value]

        # Compute predictions for the sequence.
        for t in range(1, self.recurrent_samples):
            inputs = self._get_input_data_at_t(x, t=t)
            inputs = self.set_buffers(inputs)
            y_pred = self.nss_model(inputs)
            y_pred_unpadded = self.update_buffers(inputs, y_pred)
            for key, value in y_pred_unpadded.items():
                outputs[key].append(value)

        # Convert dict values from a list of (N, C, H, W) tensors into
        # single tensors with time dimension i.e. (N, T, C, H, W).
        for key, value in outputs.items():
            outputs[key] = torch.stack(value, axis=1)

        return outputs

    def set_buffers(self, x: TensorData) -> TensorData:
        """Set or retrieve history buffers"""
        input_tensors = {}

        for key, value in x.items():
            input_tensors[key] = self._pad(value)

        # Add history buffers to input tensors.
        for key, buffer in self.history_buffers.items():
            if not buffer.initialised:
                input_tensors[buffer.name] = buffer.set(
                    input_tensors[buffer.reset_key], x["seq"]
                )
            else:
                input_tensors[buffer.name] = buffer.get(
                    input_tensors[buffer.reset_key], x["seq"]
                )

        return input_tensors

    def update_buffers(
        self, inputs: TensorData, outputs: TensorData
    ) -> dict[list[torch.tensor]]:
        """Update history buffers"""
        for _, buffer in self.history_buffers.items():
            if buffer.update_key in outputs:
                buffer.update(outputs[buffer.update_key])
            elif buffer.update_key in inputs:
                buffer.update(inputs[buffer.update_key])
            else:
                raise KeyError(
                    f"{buffer.update_key} not found in inputs or outputs of network"
                )

        # Now unpad outputs
        final_outputs = {}
        for k, v in outputs.items():
            final_outputs[k] = self._unpad(v)

        return final_outputs

    def reset_history_buffers(self):
        """Reset history buffers"""
        self.history_buffers = self.nss_model.init_history_buffers()

    def detach_buffers(self) -> None:
        """Detach history buffers"""
        for _, buffer in self.history_buffers.items():
            buffer.detach()

    def _get_pad_sz(
        self, height: int, width: int, is_unpad: bool = False
    ) -> Tuple[int, int]:
        """Return padding size - new resolutions need to be added to table"""
        padding_table = {
            # 540 -> 1080
            # -------------
            # height:
            torch.tensor(1080): torch.tensor(8),
            torch.tensor(540): torch.tensor(4),
            # 830 -> 1660
            # -------------
            # height:
            torch.tensor(830): torch.tensor(2),
            torch.tensor(1660): torch.tensor(4),
            # width:
            torch.tensor(1476): torch.tensor(4),
            torch.tensor(2952): torch.tensor(8),
        }
        # Default is no padding, unless in table
        pad_h = torch.tensor(0)
        pad_w = torch.tensor(0)
        for size, padding in padding_table.items():
            size = size + padding if is_unpad else size
            pad_h = torch.where(size == height, padding, pad_h)
            pad_w = torch.where(size == width, padding, pad_w)
        return pad_h, pad_w

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2], x.shape[3]
        pad_h, pad_w = self._get_pad_sz(height, width)
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return padded

    def _unpad(self, x: torch.Tensor) -> torch.Tensor:
        """Removes Padding"""
        if not self.unpad:
            return x
        height, width = x.shape[2], x.shape[3]
        pad_h, pad_w = self._get_pad_sz(height, width, is_unpad=True)
        return x[..., : height - pad_h, : width - pad_w]

    def _get_input_data_at_t(self, data: TensorData, t: int) -> TensorData:
        """Retrieve data at time step `t` from 5D Recurrent Tensor"""
        input_tensors = {}
        for key in data.keys():
            input_tensors[key] = data[key][:, t, :, :, :]
        return input_tensors
