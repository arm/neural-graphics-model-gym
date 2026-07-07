# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code,too-many-lines
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from pydantic import ValidationError

from ng_model_gym.core.config.config_model import NSSModelSettings
from ng_model_gym.core.data.data_utils import tonemap_forward
from ng_model_gym.core.model import BaseNGModel, create_model
from ng_model_gym.core.utils.enum_definitions import TrainEvalMode
from ng_model_gym.usecases.nss.model.model_blocks_v1 import AutoEncoderV1
from tests.base_gpu_test import BaseGPUMemoryTest
from tests.testing_utils import create_simple_params


class _NSSV1ModelTestMixin:
    """Shared NSS v1 model test helpers."""

    def _data_creator_helper(self, lr_h, lr_w, hr_h, hr_w, recurrence=None):
        """Create NSS v1 recurrent tensors for a forward pass."""

        if recurrence is None:
            recurrence = self.recurrence

        render_size = torch.zeros(self.batch_size, recurrence, 2, 1, 1)
        render_size[:, :, 0, :, :] = lr_h
        render_size[:, :, 1, :, :] = lr_w

        data = {
            "colour_linear": torch.rand(self.batch_size, recurrence, 3, lr_h, lr_w),
            "depth": torch.rand(self.batch_size, recurrence, 1, lr_h, lr_w),
            "depth_params": torch.rand(self.batch_size, recurrence, 4, lr_h, lr_w),
            "ground_truth_linear": torch.rand(
                self.batch_size, recurrence, 3, hr_h, hr_w
            ),
            "jitter": torch.zeros(self.batch_size, recurrence, 2, 1, 1),
            "motion": torch.zeros(self.batch_size, recurrence, 2, hr_h, hr_w),
            "motion_lr": torch.zeros(self.batch_size, recurrence, 2, lr_h, lr_w),
            "render_size": render_size,
            "seq": torch.ones(self.batch_size, recurrence, 1, 1, 1),
            "exposure": torch.ones(self.batch_size, recurrence, 1, 1, 1),
        }
        return {key: tensor.to(self.device) for key, tensor in data.items()}

    def _stub_core_forward(self, model):
        """Replace Slang-backed core_forward with a CPU-safe recorder."""

        captured_inputs = []

        def core_forward(inputs):
            captured_inputs.append(
                {
                    key: value.detach().clone()
                    for key, value in inputs.items()
                    if isinstance(value, torch.Tensor)
                }
            )
            output_value = float(len(captured_inputs))
            output_linear = torch.full_like(inputs["history"], output_value)
            return {
                "output": output_linear,
                "output_linear": output_linear,
                "out_filtered": output_linear,
                "temporal_params": torch.full_like(
                    inputs["temporal_params_tm1"],
                    output_value,
                ),
                "derivative": torch.full_like(
                    inputs["derivative_tm1"],
                    output_value,
                ),
            }

        model.core_forward = core_forward
        return captured_inputs

    def _init_nss_v1_model_test_state(self) -> None:
        """Set up shared NSS v1 model test state."""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = create_simple_params(usecase="nss_v1")
        self.params.model_train_eval_mode = TrainEvalMode.FP32
        self.params.train.batch_size = 2
        self.params.model.recurrent_samples = 4
        self.batch_size = self.params.train.batch_size
        self.recurrence = self.params.model.recurrent_samples


class TestNSSV1Model(  # pylint: disable=too-many-public-methods
    _NSSV1ModelTestMixin,
    unittest.TestCase,
):
    """Tests for NSS v1 model registration and public guardrails."""

    def setUp(self) -> None:
        self._init_nss_v1_model_test_state()

    def test_config_accepts_non_integer_nss_v1_scale(self) -> None:
        """NSS config accepts numeric scales greater than 1.0."""

        settings = NSSModelSettings(
            name="nss",
            model_source="prebuilt",
            version="1",
            scale=1.5,
            recurrent_samples=4,
            quality="high",
        )

        self.assertEqual(settings.scale, 1.5)

    def test_config_rejects_nss_v1_scale_at_or_below_one(self) -> None:
        """NSS config rejects scales that do not upscale."""

        for scale in (1.0, 0.75):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValidationError, "greater than 1"):
                    NSSModelSettings(
                        name="nss",
                        model_source="prebuilt",
                        version="1",
                        scale=scale,
                        recurrent_samples=4,
                        quality="high",
                    )

    def test_config_coerces_integer_nss_v1_scale(self) -> None:
        """NSS config accepts integer scales and stores them as floats."""

        settings = NSSModelSettings(
            name="nss",
            model_source="prebuilt",
            version="1",
            scale=2,
            recurrent_samples=4,
            quality="high",
        )

        self.assertEqual(settings.scale, 2.0)
        self.assertIsInstance(settings.scale, float)

    def test_config_preserves_legacy_nss_scale_contract(self) -> None:
        """Legacy NSS configs remain restricted to the existing 2x scale."""

        with self.assertRaisesRegex(ValidationError, "NSS scale must be 2.0"):
            NSSModelSettings(
                name="nss",
                model_source="prebuilt",
                version="0.1",
                scale=1.5,
                recurrent_samples=4,
                quality="high",
            )

    def test_create_nss_v1_model(self) -> None:
        """NSS v1 creates a BaseNGModel with AutoEncoderV1."""

        model = create_model(self.params, self.device)

        self.assertIsInstance(model, BaseNGModel)
        self.assertIsInstance(model.get_neural_network(), AutoEncoderV1)
        self.assertTrue(model.shader_accurate)

    def test_normalized_lr_motion_is_rejected_for_nss_v1(self) -> None:
        """NSS v1 requires preserved low-resolution motion vectors."""

        self.params.model.normalize_lr_motion = True

        for model_quality in ("high", "mid", "low"):
            with self.subTest(model_quality=model_quality):
                self.params.model.quality = model_quality

                with self.assertRaisesRegex(
                    ValueError,
                    "model.normalize_lr_motion=False",
                ):
                    create_model(self.params, self.device)

    def test_device_reports_autoencoder_device(self) -> None:
        """NSS v1 device tracks the trainable network device."""

        model = create_model(self.params, self.device)

        self.assertEqual(
            model.device,
            next(model.get_neural_network().parameters()).device,
        )

        model.get_neural_network().to(torch.device("meta"))

        self.assertEqual(
            model.device,
            next(model.get_neural_network().parameters()).device,
        )

    def test_core_forward_requires_cuda_for_slang_path(self) -> None:
        """Real NSS v1 Slang-backed forward fails clearly on CPU."""

        model = create_model(self.params, torch.device("cpu"))
        one_frame = {
            key: tensor[:, 0, :, :, :].cpu()
            for key, tensor in self._data_creator_helper(
                128, 128, 256, 256, recurrence=1
            ).items()
        }
        one_frame = model.set_buffers(one_frame)

        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            model.core_forward(one_frame)

    def test_high_quality_non_multiple_lr_shapes_use_full_res_processing_with_padded_state(
        self,
    ) -> None:
        """High quality processes at LR resolution and pads recurrent temporal state."""

        self.params.model.quality = "high"
        model = create_model(self.params, self.device)
        one_frame = {
            key: tensor[:, 0, :, :, :]
            for key, tensor in self._data_creator_helper(130, 132, 260, 264).items()
        }
        one_frame = model.set_buffers(one_frame)

        dispatch_dims = model._calculate_dispatch_dims(one_frame)
        input_shape, process_shape, hr_shape, pad_shape, depth_shape = dispatch_dims

        self.assertEqual(input_shape, (self.batch_size, 3, 130, 132))
        self.assertEqual(process_shape, (self.batch_size, 3, 130, 132))
        self.assertEqual(hr_shape, (self.batch_size, 3, 260, 264))
        self.assertEqual(pad_shape, (self.batch_size, 3, 136, 136))
        self.assertEqual(depth_shape, (self.batch_size, 1, 65, 66))
        self.assertEqual(
            one_frame["temporal_params_tm1"].shape,
            (self.batch_size, 4, 136, 136),
        )
        self.assertEqual(
            one_frame["derivative_tm1"].shape,
            (self.batch_size, 4, 130, 132),
        )

    def test_low_mid_quality_non_multiple_lr_shapes_use_half_res_processing_with_padded_state(
        self,
    ) -> None:
        """Low and mid quality process at half LR and pad recurrent temporal/derivative state."""

        for model_quality in ("low", "mid"):
            with self.subTest(model_quality=model_quality):
                self.params.model.quality = model_quality
                model = create_model(self.params, self.device)
                one_frame = {
                    key: tensor[:, 0, :, :, :]
                    for key, tensor in self._data_creator_helper(
                        130, 132, 260, 264
                    ).items()
                }
                one_frame = model.set_buffers(one_frame)

                dispatch_dims = model._calculate_dispatch_dims(one_frame)
                (
                    input_shape,
                    process_shape,
                    hr_shape,
                    pad_shape,
                    depth_shape,
                ) = dispatch_dims

                self.assertEqual(input_shape, (self.batch_size, 3, 130, 132))
                self.assertEqual(process_shape, (self.batch_size, 3, 65, 66))
                self.assertEqual(hr_shape, (self.batch_size, 3, 260, 264))
                self.assertEqual(pad_shape, (self.batch_size, 3, 72, 72))
                self.assertEqual(depth_shape, (self.batch_size, 1, 32, 33))
                self.assertEqual(
                    one_frame["temporal_params_tm1"].shape,
                    (self.batch_size, 4, 72, 72),
                )
                self.assertEqual(
                    one_frame["derivative_tm1"].shape,
                    (self.batch_size, 4, 72, 72),
                )

    def test_low_mid_quality_uses_packed_nearest_offset_channels(self) -> None:
        """Low and mid quality allocates both channels used by packed nearest offsets."""

        for model_quality in ("low", "mid"):
            with self.subTest(model_quality=model_quality):
                self.params.model.quality = model_quality
                model = create_model(self.params, self.device)

                self.assertTrue(model.packed_nearest_offset_quad)
                self.assertEqual(model._nearest_depth_offset_channels(), 2)

    def test_high_quality_uses_single_nearest_offset_channel(self) -> None:
        """High quality keeps the unpacked single-channel nearest offset encoding."""

        self.params.model.quality = "high"
        model = create_model(self.params, self.device)

        self.assertFalse(model.packed_nearest_offset_quad)
        self.assertEqual(model._nearest_depth_offset_channels(), 1)

    def test_non_integer_scale_dispatch_dims_use_rounded_output_shape(self) -> None:
        """NSS v1 dispatch dims use rounded non-integer scale output shape."""

        self.params.model.scale = 1.3
        model = create_model(self.params, self.device)
        one_frame = {
            key: tensor[:, 0, :, :, :]
            for key, tensor in self._data_creator_helper(129, 131, 168, 170).items()
        }
        one_frame = model.set_buffers(one_frame)

        _, _, hr_shape, _, depth_shape = model._calculate_dispatch_dims(one_frame)

        self.assertEqual(hr_shape, (self.batch_size, 3, 168, 170))
        self.assertEqual(depth_shape, (self.batch_size, 1, 64, 65))
        self.assertEqual(one_frame["history"].shape, (self.batch_size, 3, 168, 170))
        self.assertEqual(
            one_frame["derivative_tm1"].shape, (self.batch_size, 4, 129, 131)
        )

    def test_reset_history_shape_uses_scale_not_ground_truth(
        self,
    ) -> None:
        """Reset history uses rounded output shape not 2x GT shape."""

        self.params.model.scale = 1.3
        model = create_model(self.params, self.device)
        one_frame = {
            key: tensor[:, 0, :, :, :]
            for key, tensor in self._data_creator_helper(129, 131, 258, 262).items()
        }

        one_frame = model.set_buffers(one_frame)

        self.assertEqual(one_frame["history"].shape, (self.batch_size, 3, 168, 170))

    def test_ground_truth_linear_resizes_to_rounded_output_shape(self) -> None:
        """NSS v1 resizes 2x dataset GT to the configured output scale."""

        self.params.model.scale = 1.3
        model = create_model(self.params, self.device)
        data = self._data_creator_helper(129, 131, 258, 262, recurrence=2)
        data["ground_truth_linear"] = data["ground_truth_linear"] * 8.0
        data["exposure"] = torch.full_like(data["exposure"], 1.7)
        y_true = tonemap_forward(
            data["ground_truth_linear"] * data["exposure"],
            mode=model.tonemapper,
        )

        resized_inputs, resized_y = model.on_after_batch_transfer((data, y_true))

        expected_linear = F.interpolate(
            data["ground_truth_linear"].reshape(self.batch_size * 2, 3, 258, 262),
            size=(168, 170),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).reshape(self.batch_size, 2, 3, 168, 170)
        resized_tonemapped_y = F.interpolate(
            y_true.reshape(self.batch_size * 2, 3, 258, 262),
            size=(168, 170),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).reshape(self.batch_size, 2, 3, 168, 170)
        expected_y = tonemap_forward(
            expected_linear * data["exposure"],
            mode=model.tonemapper,
        )

        self.assertEqual(
            resized_inputs["ground_truth_linear"].shape,
            (self.batch_size, 2, 3, 168, 170),
        )
        self.assertEqual(resized_y.shape, (self.batch_size, 2, 3, 168, 170))
        torch.testing.assert_close(
            resized_inputs["ground_truth_linear"],
            expected_linear,
        )
        torch.testing.assert_close(
            resized_y,
            expected_y,
        )
        self.assertFalse(
            torch.allclose(
                resized_y,
                resized_tonemapped_y,
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_ground_truth_channel_mismatch_raises_clear_error(self) -> None:
        """NSS v1 rejects GT tensors with wrong channel count."""

        model = create_model(self.params, self.device)
        one_frame = {
            key: tensor[:, 0, :, :, :]
            for key, tensor in self._data_creator_helper(128, 128, 256, 256).items()
        }
        one_frame["ground_truth_linear"] = torch.rand(
            self.batch_size,
            1,
            256,
            256,
            device=self.device,
        )

        with self.assertRaisesRegex(
            ValueError,
            "NSS-v1 ground_truth_linear shape mismatch",
        ):
            model._validate_ground_truth_shape(
                one_frame, (self.batch_size, 3, 256, 256)
            )

    def test_core_forward_validates_ground_truth_before_loading_slang(self) -> None:
        """NSS v1 rejects bad GT shape before loading Slang."""

        model = create_model(self.params, self.device)
        one_frame = {
            key: tensor[:, 0, :, :, :]
            for key, tensor in self._data_creator_helper(128, 128, 256, 256).items()
        }
        one_frame = model.set_buffers(one_frame)
        one_frame["ground_truth_linear"] = torch.rand(
            self.batch_size,
            1,
            256,
            256,
            device=self.device,
        )

        with patch.object(model, "_get_slang", return_value=object()) as get_slang:
            with self.assertRaisesRegex(
                ValueError,
                "NSS-v1 ground_truth_linear shape mismatch",
            ):
                model.core_forward(one_frame)

        get_slang.assert_not_called()

    def test_high_quality_offset_lut_uses_compact_metadata_layout(self) -> None:
        """High quality passes only idx modulo metadata for rounded LR/HR shapes."""

        self.params.model.scale = 1.5
        self.params.model.quality = "high"
        model = create_model(self.params, self.device)
        jitter = torch.zeros(self.batch_size, 2, 1, 1, device=self.device)

        offset_lut, idx_modulo = model._generate_offset_lut(
            jitter,
            in_shape=(self.batch_size, 3, 8, 10),
            out_shape=(self.batch_size, 3, 12, 15),
        )

        self.assertEqual(offset_lut.shape, (self.batch_size, 6, 9, 9))
        torch.testing.assert_close(
            idx_modulo,
            torch.tensor([[[[3.0, 3.0]]]], device=self.device),
            rtol=0,
            atol=0,
        )

    def test_low_mid_quality_offset_lut_uses_post_process_metadata_layout(
        self,
    ) -> None:
        """Low and mid quality pass idx modulo and preprocess dimensions by channel."""
        for model_quality in ("low", "mid"):
            with self.subTest(model_quality=model_quality):
                self.params.model.scale = 1.5
                self.params.model.quality = model_quality
                model = create_model(self.params, self.device)
                jitter = torch.zeros(self.batch_size, 2, 1, 1, device=self.device)

                offset_lut, idx_modulo = model._generate_offset_lut(
                    jitter,
                    in_shape=(self.batch_size, 3, 8, 10),
                    out_shape=(self.batch_size, 3, 12, 15),
                )

                self.assertEqual(offset_lut.shape, (self.batch_size, 6, 9, 4))

                # Low- and mid-quality use half-resolution, turning 8 and 10
                # (in_shape, above) into 4.0 and 5.0.
                expected = torch.tensor(
                    [3.0, 3.0, 4.0, 5.0],
                    device=self.device,
                ).reshape(1, 4, 1, 1)
                torch.testing.assert_close(
                    idx_modulo,
                    expected.expand(self.batch_size, -1, -1, -1),
                    rtol=0,
                    atol=0,
                )

    def test_common_slang_defines(self) -> None:
        """
        Test Slang defines which should be used at all quality levels. Assume NSS v1.
        """

        expected_filter_kernel_taps = {"low": 4, "mid": 4, "high": 9}

        for model_quality, expected_taps in expected_filter_kernel_taps.items():
            with self.subTest(model_quality=model_quality):
                self.params.model.quality = model_quality
                model = create_model(self.params, self.device)
                with patch(
                    "ng_model_gym.usecases.nss.model.model_v1.load_slang_module",
                    return_value=object(),
                ) as load_module:
                    model._get_slang()

                defines = load_module.call_args.kwargs["defines"]
                self.assertEqual(defines["NSS_QUALITY_LOW"], 0)
                self.assertEqual(defines["NSS_QUALITY_MEDIUM"], 1)
                self.assertEqual(defines["NSS_QUALITY_HIGH"], 2)
                self.assertEqual(defines["FILTER_COLOR_KERNEL_SZ"], expected_taps)
                self.assertEqual(defines["NSS_V1_LUMA_DERIVATIVE"], 1)
                self.assertEqual(defines["NSS_V1_SHARP_THETA"], 1)

    def test_high_quality_slang_defines(self) -> None:
        """Test Slang defines which should be used for high quality."""

        self.params.model.quality = "high"
        model = create_model(self.params, self.device)
        with patch(
            "ng_model_gym.usecases.nss.model.model_v1.load_slang_module",
            return_value=object(),
        ) as load_module:
            model._get_slang()

        defines = load_module.call_args.kwargs["defines"]
        self.assertEqual(defines["NSS_QUALITY"], 2)  # 2 = "high"
        self.assertEqual(defines["NSS_PREPROCESS_HALF_RES_INPUT"], False)
        self.assertEqual(defines["NSS_DEPTH_SCATTER_QUARTER_RES_INPUT"], False)
        self.assertEqual(defines["NSS_USE_SPARSE_2X2_FILTER"], False)
        self.assertEqual(defines["NSS_USE_HISTORY_CATMULL"], True)
        self.assertEqual(defines["NSS_PACKED_NEAREST_OFFSET_QUAD"], False)

    def test_mid_quality_slang_defines(self) -> None:
        """Test Slang defines which should be used for mid quality."""

        self.params.model.quality = "mid"
        model = create_model(self.params, self.device)
        with patch(
            "ng_model_gym.usecases.nss.model.model_v1.load_slang_module",
            return_value=object(),
        ) as load_module:
            model._get_slang()

        defines = load_module.call_args.kwargs["defines"]
        self.assertEqual(defines["NSS_QUALITY"], 1)  # 1 = "mid"
        self.assertEqual(defines["NSS_PREPROCESS_HALF_RES_INPUT"], True)
        self.assertEqual(defines["NSS_DEPTH_SCATTER_QUARTER_RES_INPUT"], True)
        self.assertEqual(defines["NSS_USE_SPARSE_2X2_FILTER"], True)
        self.assertEqual(defines["NSS_USE_HISTORY_CATMULL"], True)
        self.assertEqual(defines["NSS_PACKED_NEAREST_OFFSET_QUAD"], True)

    def test_low_quality_slang_defines(self) -> None:
        """Test Slang defines which should be used for low quality."""

        self.params.model.quality = "low"
        model = create_model(self.params, self.device)
        with patch(
            "ng_model_gym.usecases.nss.model.model_v1.load_slang_module",
            return_value=object(),
        ) as load_module:
            model._get_slang()

        defines = load_module.call_args.kwargs["defines"]
        self.assertEqual(defines["NSS_QUALITY"], 0)  # 1 = "low"
        self.assertEqual(defines["NSS_PREPROCESS_HALF_RES_INPUT"], True)
        self.assertEqual(defines["NSS_DEPTH_SCATTER_QUARTER_RES_INPUT"], True)
        self.assertEqual(defines["NSS_USE_SPARSE_2X2_FILTER"], True)
        self.assertEqual(defines["NSS_USE_HISTORY_CATMULL"], False)
        self.assertEqual(defines["NSS_PACKED_NEAREST_OFFSET_QUAD"], True)

    def test_forward_uses_available_time_dimension(self) -> None:
        """NSS v1 does not index past the input recurrent dimension."""

        self.params.model.recurrent_samples = 16
        model = create_model(self.params, self.device)
        captured_inputs = self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=1)

        model_out = model(data)

        self.assertEqual(len(captured_inputs), 1)
        self.assertEqual(model_out["output_linear"].shape[1], 1)

    def test_split_inputs_over_time_matches_select_indexing(self) -> None:
        """Pre-split per-frame inputs match select-based indexing."""

        model = create_model(self.params, self.device)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=3)

        frames = model._split_inputs_over_time(data, sequence_length=3)

        self.assertEqual(len(frames), 3)
        for t, frame in enumerate(frames):
            self.assertEqual(frame.keys(), data.keys())
            for key, value in frame.items():
                torch.testing.assert_close(
                    value,
                    data[key].select(dim=1, index=t),
                    rtol=0,
                    atol=0,
                    msg=key,
                )

    def test_split_inputs_over_time_splits_rank_two_and_higher_tensors(self) -> None:
        """Split tensors with a long-enough time dimension."""

        model = create_model(self.params, self.device)
        data = {
            "colour_linear": torch.arange(
                self.batch_size * 2 * 3 * 4 * 4,
                device=self.device,
                dtype=torch.float32,
            ).reshape(self.batch_size, 2, 3, 4, 4),
            "rank2_metadata": torch.arange(
                self.batch_size * 2,
                device=self.device,
                dtype=torch.float32,
            ).reshape(self.batch_size, 2),
            "rank3_metadata": torch.arange(
                self.batch_size * 2 * 3,
                device=self.device,
                dtype=torch.float32,
            ).reshape(self.batch_size, 2, 3),
            "rank4_metadata": torch.arange(
                self.batch_size * 2 * 3 * 4,
                device=self.device,
                dtype=torch.float32,
            ).reshape(self.batch_size, 2, 3, 4),
        }

        frames = model._split_inputs_over_time(data, sequence_length=2)

        self.assertEqual(len(frames), 2)
        for frame_idx, frame in enumerate(frames):
            for key, value in frame.items():
                torch.testing.assert_close(
                    value,
                    data[key].select(dim=1, index=frame_idx),
                    rtol=0,
                    atol=0,
                    msg=key,
                )

    def test_forward_exposes_loss_context_keys(self) -> None:
        """NSS v1 recurrent output includes loss context tensors."""

        model = create_model(self.params, self.device)
        self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=2)
        data["motion"][:, 0, ...] = 5.0
        data["motion"][:, 1, ...] = 7.0

        model_out = model(data)

        self.assertIn("motion", model_out)
        self.assertIn("reset_event", model_out)
        self.assertIs(model_out["motion"], data["motion"])
        self.assertEqual(model_out["motion"].shape, data["motion"].shape)
        self.assertEqual(model_out["reset_event"].shape, data["seq"].shape)
        torch.testing.assert_close(
            model_out["motion"][:, 0, ...],
            torch.full_like(model_out["motion"][:, 0, ...], 5.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model_out["motion"][:, 1, ...],
            torch.full_like(model_out["motion"][:, 1, ...], 7.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model_out["reset_event"][:, 0, ...],
            torch.zeros_like(model_out["reset_event"][:, 0, ...]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model_out["reset_event"][:, 1, ...],
            torch.ones_like(model_out["reset_event"][:, 1, ...]),
            rtol=0,
            atol=0,
        )

    def test_forward_overwrites_loss_context_keys(self) -> None:
        """NSS v1 recurrent output ignores core-owned loss context tensors."""

        model = create_model(self.params, self.device)

        def core_forward(inputs):
            output_linear = torch.zeros_like(inputs["history"])
            return {
                "output": output_linear,
                "output_linear": output_linear,
                "out_filtered": output_linear,
                "temporal_params": torch.zeros_like(inputs["temporal_params_tm1"]),
                "derivative": torch.zeros_like(inputs["derivative_tm1"]),
                "motion": torch.full_like(inputs["motion"], -99.0),
                "reset_event": torch.full_like(inputs["reset_event"], -42.0),
            }

        model.core_forward = core_forward
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=2)
        data["motion"][:, 0, ...] = 5.0
        data["motion"][:, 1, ...] = 7.0

        model_out = model(data)

        self.assertIs(model_out["motion"], data["motion"])
        torch.testing.assert_close(
            model_out["motion"][:, 0, ...],
            torch.full_like(model_out["motion"][:, 0, ...], 5.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model_out["motion"][:, 1, ...],
            torch.full_like(model_out["motion"][:, 1, ...], 7.0),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model_out["reset_event"][:, 0, ...],
            torch.zeros_like(model_out["reset_event"][:, 0, ...]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model_out["reset_event"][:, 1, ...],
            torch.ones_like(model_out["reset_event"][:, 1, ...]),
            rtol=0,
            atol=0,
        )

    def test_forward_preserves_history_across_same_sequence_calls(self) -> None:
        """NSS v1 keeps recurrent history across forward calls with matching seq."""

        model = create_model(self.params, self.device)
        captured_inputs = self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=1)

        model(data)
        model(data)

        self.assertTrue(torch.all(captured_inputs[0]["history"] == 0))
        self.assertTrue(torch.all(captured_inputs[1]["history"] == 1))

    def test_forward_resets_history_when_sequence_changes(self) -> None:
        """NSS v1 resets recurrent history when seq changes."""

        model = create_model(self.params, self.device)
        captured_inputs = self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=1)
        next_sequence_data = self._data_creator_helper(
            128,
            128,
            256,
            256,
            recurrence=1,
        )
        next_sequence_data["seq"] = torch.full_like(next_sequence_data["seq"], 2.0)

        model(data)
        model(next_sequence_data)

        self.assertTrue(torch.all(captured_inputs[0]["history"] == 0))
        self.assertTrue(torch.all(captured_inputs[1]["history"] == 0))

    def test_forward_zeroes_motion_on_initial_and_reset_frames(self) -> None:
        """NSS v1 zeroes motion inputs when recurrent state is reset."""

        model = create_model(self.params, self.device)
        captured_inputs = self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=1)
        data["motion"].fill_(7.0)
        data["motion_lr"].fill_(3.0)
        next_sequence_data = self._data_creator_helper(
            128,
            128,
            256,
            256,
            recurrence=1,
        )
        next_sequence_data["motion"].fill_(7.0)
        next_sequence_data["motion_lr"].fill_(3.0)
        next_sequence_data["seq"] = torch.full_like(next_sequence_data["seq"], 2.0)

        model(data)
        model(data)
        model(next_sequence_data)

        self.assertTrue(torch.all(captured_inputs[0]["motion"] == 0))
        self.assertTrue(torch.all(captured_inputs[0]["motion_lr"] == 0))
        self.assertTrue(torch.all(captured_inputs[1]["motion"] == 7))
        self.assertTrue(torch.all(captured_inputs[1]["motion_lr"] == 3))
        self.assertTrue(torch.all(captured_inputs[2]["motion"] == 0))
        self.assertTrue(torch.all(captured_inputs[2]["motion_lr"] == 0))

    def test_gt_history_augmentation_uses_target_space_history_during_training(
        self,
    ) -> None:
        """GT history augmentation seeds first-frame history from loss target."""

        self.params.model.gt_history_augmentation = True
        self.params.model.gt_history_augmentation_chance = 100.0
        model = create_model(self.params, self.device)
        model.train()
        captured_inputs = self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=2)
        data["ground_truth_linear"].fill_(0.25)
        y_true = torch.full(
            (self.batch_size, self.recurrence, 3, 256, 256),
            0.75,
            device=self.device,
        )
        model.y_true = y_true

        model(data)

        torch.testing.assert_close(
            captured_inputs[0]["history"],
            y_true[:, 0, ...],
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.all(captured_inputs[0]["reset_event"] == 1.0))
        self.assertTrue(torch.all(captured_inputs[1]["history"] == 1.0))

    def test_gt_history_augmentation_chance_zero_preserves_reset_history(self) -> None:
        """Chance zero leaves the reset history untouched."""

        self.params.model.gt_history_augmentation = True
        self.params.model.gt_history_augmentation_chance = 0.0
        model = create_model(self.params, self.device)
        model.train()
        captured_inputs = self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=1)
        model.y_true = torch.ones(
            self.batch_size,
            self.recurrence,
            3,
            256,
            256,
            device=self.device,
        )

        model(data)

        self.assertTrue(torch.all(captured_inputs[0]["history"] == 0.0))
        self.assertTrue(torch.all(captured_inputs[0]["reset_event"] == 0.0))

    def test_gt_history_augmentation_is_training_only(self) -> None:
        """GT history augmentation is disabled in eval mode."""

        self.params.model.gt_history_augmentation = True
        self.params.model.gt_history_augmentation_chance = 100.0
        model = create_model(self.params, self.device)
        model.eval()
        captured_inputs = self._stub_core_forward(model)
        data = self._data_creator_helper(128, 128, 256, 256, recurrence=1)
        model.y_true = torch.ones(
            self.batch_size,
            self.recurrence,
            3,
            256,
            256,
            device=self.device,
        )

        model(data)

        self.assertTrue(torch.all(captured_inputs[0]["history"] == 0.0))


@unittest.skipUnless(
    torch.cuda.is_available(),
    "NSS v1 forward requires CUDA because the forward path is Slang-backed.",
)
class TestNSSV1ModelGPU(_NSSV1ModelTestMixin, BaseGPUMemoryTest):
    """CUDA-backed NSS v1 model tests."""

    def setUp(self) -> None:
        super().setUp()
        self._init_nss_v1_model_test_state()
        self.device = torch.device("cuda")

    def test_shape_nss_v1_high_quality_model_forward_pass(self) -> None:
        """Test NSS v1 recurrent high-quality forward output shapes."""

        model = create_model(self.params, self.device)
        data = self._data_creator_helper(128, 128, 256, 256)

        with torch.no_grad():
            model.train()
            model_out = model(data)

        self.assertEqual(
            model_out["output_linear"].shape,
            (self.batch_size, self.recurrence, 3, 256, 256),
        )
        self.assertEqual(
            model_out["output"].shape,
            (self.batch_size, self.recurrence, 3, 256, 256),
        )
        self.assertEqual(
            model_out["out_filtered"].shape,
            (self.batch_size, self.recurrence, 3, 256, 256),
        )
        self.assertEqual(
            model_out["temporal_params"].shape,
            (self.batch_size, self.recurrence, 4, 128, 128),
        )
        self.assertEqual(
            model_out["derivative"].shape,
            (self.batch_size, self.recurrence, 4, 128, 128),
        )

    def test_shape_nss_v1_low_mid_quality_model_forward_pass(self) -> None:
        """Test NSS v1 recurrent low-/mid-quality forward output shapes."""

        for model_quality in ("low", "mid"):
            with self.subTest(model_quality=model_quality):
                self.params.model.quality = model_quality
                model = create_model(self.params, self.device)
                data = self._data_creator_helper(128, 128, 256, 256)

                with torch.no_grad():
                    model.train()
                    model_out = model(data)

                self.assertEqual(
                    model_out["output_linear"].shape,
                    (self.batch_size, self.recurrence, 3, 256, 256),
                )
                self.assertEqual(
                    model_out["output"].shape,
                    (self.batch_size, self.recurrence, 3, 256, 256),
                )
                self.assertEqual(
                    model_out["out_filtered"].shape,
                    (self.batch_size, self.recurrence, 3, 256, 256),
                )
                self.assertEqual(
                    model_out["temporal_params"].shape,
                    (self.batch_size, self.recurrence, 4, 64, 64),
                )
                self.assertEqual(
                    model_out["derivative"].shape,
                    (self.batch_size, self.recurrence, 4, 64, 64),
                )

    def test_shape_nss_v1_high_quality_non_integer_scale_forward_pass(self) -> None:
        """Test NSS v1 recurrent high-quality forward shapes for non-integer scale."""

        self.params.model.scale = 1.5
        model = create_model(self.params, self.device)
        data = self._data_creator_helper(96, 80, 144, 120)

        with torch.no_grad():
            model.eval()
            model_out = model(data)

        self.assertEqual(
            model_out["output_linear"].shape,
            (self.batch_size, self.recurrence, 3, 144, 120),
        )
        self.assertEqual(
            model_out["output"].shape,
            (self.batch_size, self.recurrence, 3, 144, 120),
        )
        self.assertEqual(
            model_out["out_filtered"].shape,
            (self.batch_size, self.recurrence, 3, 144, 120),
        )
        self.assertEqual(
            model_out["temporal_params"].shape,
            (self.batch_size, self.recurrence, 4, 96, 80),
        )
        self.assertEqual(
            model_out["derivative"].shape,
            (self.batch_size, self.recurrence, 4, 96, 80),
        )

    def test_shape_nss_v1_low_mid_quality_non_integer_scale_forward_pass(self) -> None:
        """Test NSS v1 recurrent low-/mid-quality forward shapes for non-integer scale."""

        for model_quality in ("low", "mid"):
            with self.subTest(model_quality=model_quality):
                self.params.model.quality = model_quality
                self.params.model.scale = 1.5
                model = create_model(self.params, self.device)
                data = self._data_creator_helper(96, 80, 144, 120)

                with torch.no_grad():
                    model.eval()
                    model_out = model(data)

                self.assertEqual(
                    model_out["output_linear"].shape,
                    (self.batch_size, self.recurrence, 3, 144, 120),
                )
                self.assertEqual(
                    model_out["output"].shape,
                    (self.batch_size, self.recurrence, 3, 144, 120),
                )
                self.assertEqual(
                    model_out["out_filtered"].shape,
                    (self.batch_size, self.recurrence, 3, 144, 120),
                )
                self.assertEqual(
                    model_out["temporal_params"].shape,
                    (self.batch_size, self.recurrence, 4, 48, 40),
                )
                self.assertEqual(
                    model_out["derivative"].shape,
                    (self.batch_size, self.recurrence, 4, 48, 40),
                )
