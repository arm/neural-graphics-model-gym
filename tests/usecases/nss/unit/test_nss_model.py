# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from typing import NamedTuple, Tuple

import torch

from ng_model_gym.core.model import BaseNGModel, create_model
from ng_model_gym.core.utils.types import TrainEvalMode
from ng_model_gym.usecases.nss.model.model_blocks import AutoEncoderV1
from tests.base_gpu_test import BaseGPUMemoryTest
from tests.testing_utils import create_simple_params


class TestNSS(BaseGPUMemoryTest):
    """Tests for the NSS model"""

    def _data_creator_helper(self, lr_h, lr_w, hr_h, hr_w):
        data = {
            "colour_linear": torch.randn(self.batch, self.recurrence, 3, lr_h, lr_w),
            "depth": torch.randn(self.batch, self.recurrence, 1, lr_h, lr_w),
            "depth_params": torch.randn(self.batch, self.recurrence, 4, lr_h, lr_w),
            "ground_truth_linear": torch.randn(
                self.batch, self.recurrence, 3, hr_h, hr_w
            ),
            "jitter": torch.randn(self.batch, self.recurrence, 2, 1, 1),
            "motion": torch.randn(self.batch, self.recurrence, 2, hr_h, hr_w),
            "render_size": torch.randn(self.batch, self.recurrence, 2, 1, 1),
            "zNear": torch.randn(self.batch, self.recurrence, 1, 1, 1),
            "zFar": torch.randn(self.batch, self.recurrence, 1, 1, 1),
            "seq": torch.randn(self.batch, self.recurrence, 1, 1, 1),
            "exposure": torch.randn(self.batch, self.recurrence, 1, 1, 1),
            "colour": torch.randn(self.batch, self.recurrence, 3, lr_h, lr_w),
        }
        tensor_dict = {key: tensor.to(self.device) for key, tensor in data.items()}
        return tensor_dict

    def setUp(self):
        """Setup NSS model."""
        super().setUp()
        params = create_simple_params(dataset="")
        params.model_train_eval_mode = TrainEvalMode.FP32
        params.dataset.gt_augmentation = True
        params.train.batch_size = 2
        params.dataset.recurrent_samples = 4
        self.device = torch.device("cuda")
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        self.model = create_model(params, self.device)
        if not isinstance(self.model, BaseNGModel):
            raise TypeError("Model is not a BaseNGModel")
        self.batch = params.train.batch_size
        self.recurrence = params.dataset.recurrent_samples
        self.data = self._data_creator_helper(128, 128, 256, 256)
        self.params = params

    def test_shape_nss_model_forward_pass(self):
        """Test nss model training shape"""
        with torch.no_grad():
            self.model.train()
            model_out = self.model(self.data)

        self.assertEqual(
            model_out["output_linear"].shape, (self.batch, self.recurrence, 3, 256, 256)
        )
        self.assertEqual(
            model_out["output"].shape, (self.batch, self.recurrence, 3, 256, 256)
        )
        self.assertEqual(
            model_out["feedback"].shape, (self.batch, self.recurrence, 4, 128, 128)
        )
        self.assertEqual(
            model_out["derivative"].shape, (self.batch, self.recurrence, 2, 128, 128)
        )
        self.assertEqual(
            model_out["depth_dilated"].shape, (self.batch, self.recurrence, 1, 128, 128)
        )

    def test_golden_nss_model_forward(self):
        """Test nss model training"""

        nss_input_golden = torch.load(
            "tests/usecases/nss/unit/data/nss_v1_golden_values/nss_input_golden.pt",
            map_location=self.device,
            weights_only=True,
        )

        nss_output_golden = torch.load(
            "tests/usecases/nss/unit/data/nss_v1_golden_values/nss_output_golden.pt",
            map_location=self.device,
            weights_only=True,
        )["outputs"]

        self.model.train()

        autoencoder_with_golden_state = AutoEncoderV1()
        autoencoder_with_golden_state.load_state_dict(
            nss_input_golden["autoencoder_state"]
        )
        autoencoder_with_golden_state.to(self.device)
        self.model.set_neural_network(autoencoder_with_golden_state)

        model_out = self.model(nss_input_golden["feedback_input"])

        tolerance = 1e-3
        expected_output_linear = nss_output_golden["output_linear"]
        torch.testing.assert_close(
            model_out["output_linear"],
            expected_output_linear,
            rtol=tolerance,
            atol=tolerance,
        )

        expected_output = nss_output_golden["output"]
        self.assertTrue(
            torch.allclose(
                model_out["output"], expected_output, rtol=tolerance, atol=tolerance
            )
        )

        expected_feedback = nss_output_golden["feedback"]
        torch.testing.assert_close(
            model_out["feedback"], expected_feedback, rtol=tolerance, atol=tolerance
        )

        expected_derivative = nss_output_golden["derivative"]
        torch.testing.assert_close(
            model_out["derivative"],
            expected_derivative,
            rtol=tolerance,
            atol=tolerance,
        )

        expected_depth_dilated = nss_output_golden["depth_dilated"]
        torch.testing.assert_close(
            model_out["depth_dilated"],
            expected_depth_dilated,
            rtol=tolerance,
            atol=tolerance,
        )

    def test_pad_sz_and_unpad(self):
        """Check padding and unpadding for a variety of resolutions"""

        class PaddingTest(NamedTuple):
            """Namedtuple for a padding test"""

            lr: Tuple[int, int]
            expected_lr_padding: Tuple[int, int]
            hr: Tuple[int, int]
            expected_hr_padding: Tuple[int, int]

        padding_tests = [
            PaddingTest(
                lr=(128, 128),
                expected_lr_padding=(0, 0),
                hr=(256, 256),
                expected_hr_padding=(0, 0),
            ),
            PaddingTest(
                lr=(540, 960),
                expected_lr_padding=(4, 0),
                hr=(1080, 1920),
                expected_hr_padding=(8, 0),
            ),
            PaddingTest(
                lr=(256, 256),
                expected_lr_padding=(0, 0),
                hr=(512, 512),
                expected_hr_padding=(0, 0),
            ),
            PaddingTest(
                lr=(540, 738),
                expected_lr_padding=(4, 6),
                hr=(1080, 1476),
                expected_hr_padding=(8, 12),
            ),
            PaddingTest(
                lr=(830, 1476),
                expected_lr_padding=(2, 4),
                hr=(1660, 2952),
                expected_hr_padding=(4, 8),
            ),
            PaddingTest(
                lr=(101, 309),
                expected_lr_padding=(3, 3),
                hr=(202, 618),
                expected_hr_padding=(6, 6),
            ),
        ]
        for padding_test in padding_tests:
            # Create new nss model
            nss_model = create_model(self.params, self.device)

            # Create dataset
            sample_input_data = self._data_creator_helper(
                *padding_test.lr, *padding_test.hr
            )
            # 4 dims for the padding input tensors
            sample_input_data = {k: v[0] for k, v in sample_input_data.items()}

            # Manually create padding policy as we are not running the NSSModel forward pass
            padding_policy = nss_model.create_padding_policy(sample_input_data)
            nss_model.padding_policy = padding_policy

            # Check padding policy padding calculations are correct
            self.assertEqual(padding_policy.hr, padding_test.hr)
            self.assertEqual(padding_policy.lr, padding_test.lr)

            self.assertEqual(
                padding_policy.lr_padding, padding_test.expected_lr_padding
            )
            self.assertEqual(
                padding_policy.hr_padding, padding_test.expected_hr_padding
            )

            nss_model.padding_policy = padding_policy

            padded_tensors = []

            # Iterate over input tensors and test padding
            for tensor in sample_input_data.values():
                height, width = tensor.shape[2], tensor.shape[3]
                pad_h, pad_w = nss_model._get_pad_sz(height, width, is_unpad=False)
                pad_h, pad_w = pad_h.item(), pad_w.item()

                match (height, width):
                    case padding_test.lr:
                        self.assertEqual(
                            (pad_h, pad_w), padding_test.expected_lr_padding
                        )
                    case padding_test.hr:
                        self.assertEqual(
                            (pad_h, pad_w), padding_test.expected_hr_padding
                        )
                    case (1, 1):
                        self.assertEqual((pad_h, pad_w), (0, 0))
                    case _:
                        raise ValueError("Unexpected height/width")

                padded_tensors.append(
                    torch.nn.functional.pad(
                        tensor, (0, pad_w, 0, pad_h), mode="reflect"
                    )
                )

            # Iterate over padded_tensors and test unpadding
            for tensor in padded_tensors:
                height, width = tensor.shape[2], tensor.shape[3]
                pad_h, pad_w = nss_model._get_pad_sz(height, width, is_unpad=True)
                pad_h, pad_w = pad_h.item(), pad_w.item()

                # Unpad and check if we match expectations
                height -= pad_h
                width -= pad_w

                match (height, width):
                    case padding_test.lr:
                        self.assertEqual(
                            (pad_h, pad_w), padding_test.expected_lr_padding
                        )
                    case padding_test.hr:
                        self.assertEqual(
                            (pad_h, pad_w), padding_test.expected_hr_padding
                        )
                    case (1, 1):
                        self.assertEqual((pad_h, pad_w), (0, 0))
                    case _:
                        raise ValueError(f"Unexpected height/width {height=} {width=}")
