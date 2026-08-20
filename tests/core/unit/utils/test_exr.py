# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import OpenEXR
import torch

from ng_model_gym.core.utils.io.exr import write_rgb_float32_exr


class TestWriteRGBFloat32EXR(unittest.TestCase):
    """Test writing unmodified float32 RGB OpenEXR frames."""

    @patch("ng_model_gym.core.utils.io.exr.OpenEXR.File")
    def test_uses_openexr_file_context_manager(self, openexr_file) -> None:
        """Enter and exit the OpenEXR file object used for writing."""
        path = Path("frame.exr")
        frame = torch.zeros(3, 2, 2)
        entered_file = openexr_file.return_value.__enter__.return_value

        write_rgb_float32_exr(path, frame)

        openexr_file.return_value.__enter__.assert_called_once_with()
        entered_file.write.assert_called_once_with(str(path))
        openexr_file.return_value.__exit__.assert_called_once()

    def test_round_trips_both_layouts_as_float32_rgb(self) -> None:
        """Round-trip CHW and NCHW RGB frames with negative and HDR values."""
        base_frame = torch.tensor(
            [
                [[-2.0, 0.25], [1.0, 12000.0]],
                [[3.0, -0.5], [8.0, 24000.0]],
                [[5.0, 0.75], [16.0, 48000.0]],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        layouts = {
            "chw": base_frame,
            "nchw": base_frame.unsqueeze(0),
        }
        expected = base_frame.detach().cpu().to(torch.float32).numpy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            for name, frame in layouts.items():
                with self.subTest(layout=name):
                    path = Path(tmp_dir, f"{name}.exr")
                    write_rgb_float32_exr(path, frame)

                    with OpenEXR.File(str(path), separate_channels=True) as exr_file:
                        channels = exr_file.channels()
                        self.assertEqual(set(channels), {"R", "G", "B"})
                        for index, channel_name in enumerate("RGB"):
                            channel = channels[channel_name]
                            self.assertEqual(channel.type(), OpenEXR.FLOAT)
                            self.assertEqual(channel.pixels.dtype, np.float32)
                            np.testing.assert_array_equal(
                                channel.pixels,
                                expected[index],
                            )

    def test_rejects_invalid_inputs_without_creating_output(self) -> None:
        """Reject every per-frame contract violation before writing a file."""
        cases = (
            ("non_tensor", object(), TypeError, "must be a torch.Tensor"),
            (
                "wrong_rank",
                torch.zeros(3, 2),
                ValueError,
                "CHW or single-frame NCHW",
            ),
            (
                "batch",
                torch.zeros(2, 3, 2, 2),
                ValueError,
                "batch size one",
            ),
            (
                "channels",
                torch.zeros(4, 2, 2),
                ValueError,
                "three RGB channels",
            ),
            (
                "integer",
                torch.zeros(3, 2, 2, dtype=torch.int32),
                TypeError,
                "floating point dtype",
            ),
            (
                "zero_height",
                torch.zeros(3, 0, 2),
                ValueError,
                "positive height and width",
            ),
            (
                "zero_width",
                torch.zeros(1, 3, 2, 0),
                ValueError,
                "positive height and width",
            ),
            (
                "nan",
                torch.full((3, 2, 2), float("nan")),
                ValueError,
                "only finite values",
            ),
            (
                "inf",
                torch.full((3, 2, 2), float("inf")),
                ValueError,
                "only finite values",
            ),
            (
                "float32_overflow",
                torch.full((3, 2, 2), 1e300, dtype=torch.float64),
                ValueError,
                "finite when represented as float32",
            ),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            for name, frame, exception_type, message in cases:
                with self.subTest(name=name):
                    path = Path(tmp_dir, f"{name}.exr")
                    with self.assertRaisesRegex(exception_type, message):
                        write_rgb_float32_exr(path, frame)  # type: ignore[arg-type]

                    self.assertFalse(path.exists())
