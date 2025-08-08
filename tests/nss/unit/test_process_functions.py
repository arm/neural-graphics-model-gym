# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from ng_model_gym.nss.dataloader.process_functions import process_nss_data
from ng_model_gym.nss.dataloader.utils import DataLoaderMode


class TestProcessFunctions(unittest.TestCase):
    """Tests for processing functions."""

    def test_process_nss_data_output_shape(self):
        """Test output shape of process function"""
        batch_size = 16
        colour_linear = torch.rand(batch_size, 3, 128, 128)
        depth = torch.rand(batch_size, 1, 128, 128)
        depth_params = torch.rand(batch_size, 4)
        exposure = torch.rand(batch_size, 1)
        ground_truth_linear = torch.rand(batch_size, 3, 256, 256)
        jitter = torch.rand(batch_size, 2, 1, 1)
        motion = torch.rand(batch_size, 2, 256, 256)
        render_size = torch.rand(batch_size, 2)
        z_near = torch.rand(batch_size, 1)
        z_far = torch.rand(batch_size, 1)

        inputs = {
            "colour_linear": colour_linear,
            "depth": depth,
            "depth_params": depth_params,
            "exposure": exposure,
            "ground_truth_linear": ground_truth_linear,
            "jitter": jitter,
            "motion": motion,
            "render_size": render_size,
            "zNear": z_near,
            "zFar": z_far,
        }

        data_out, target = process_nss_data(inputs, DataLoaderMode.TRAIN)
        self.assertEqual(len(data_out), 11)
        self.assertEqual(target.shape, (16, 3, 256, 256))

        expected_shape = {
            "colour_linear": (16, 3, 128, 128),
            "depth": (16, 1, 128, 128),
            "depth_params": (16, 4, 1, 1),
            "ground_truth_linear": (16, 3, 256, 256),
            "jitter": (16, 2, 1, 1),
            "motion": (16, 2, 256, 256),
            "render_size": (16, 2, 1, 1),
            "zFar": (16, 1, 1, 1),
            "zNear": (16, 1, 1, 1),
            "exposure": (16, 1, 1, 1),
            "colour": (16, 3, 128, 128),
        }

        for name, tensor in data_out.items():
            self.assertEqual(expected_shape[name], tensor.shape)


if __name__ == "__main__":
    unittest.main()
