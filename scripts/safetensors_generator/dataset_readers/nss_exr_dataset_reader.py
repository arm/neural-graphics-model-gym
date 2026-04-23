# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from ng_model_gym.core.data import tonemap_inverse
from ng_model_gym.core.model.graphics_utils import swizzle
from scripts.safetensors_generator.dataset_readers.safetensors_feature_iterator import (
    EXRDatasetReader,
)
from scripts.safetensors_generator.exr_utils import read_exr_torch
from scripts.safetensors_generator.fsr2_methods import depth_to_view_space_params


class NSSEXRDatasetReader(EXRDatasetReader):
    """Dataset reader for NSS and EXR files v1.0.1 spec."""

    @staticmethod
    def _validate_exr_dataset_structure(
        src_root: Path, seq_path: Path, scale: float
    ) -> None:
        """Check the dataset is in the expected format."""
        scale_str = str(float(scale)).replace(".", "_").replace("_0", "")

        required_dirs = [
            src_root / "ground_truth" / seq_path,
            src_root / "motion_gt" / seq_path,
            src_root / f"x{scale_str}" / "color" / seq_path,
            src_root / f"x{scale_str}" / "depth" / seq_path,
            src_root / f"x{scale_str}" / "motion" / seq_path,
        ]

        for required_dir in required_dirs:
            if not required_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing {required_dir} in expected structure."
                )
            if not any(required_dir.iterdir()):
                raise ValueError(f"{required_dir} is empty.")

    @staticmethod
    def _create_depth_params(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create the depth parameters"""
        make_image_like = lambda t: t.unsqueeze(-1).unsqueeze(-1)
        depth_params = depth_to_view_space_params(
            zNear=make_image_like(data["zNear"]),
            zFar=make_image_like(data["zFar"]),
            FovY=make_image_like(data["FovY"]),
            infinite=make_image_like(data["infinite_zFar"]).to(torch.bool),
            renderSizeWidth=make_image_like(data["render_size"])[:, 1:2, ...],
            renderSizeHeight=make_image_like(data["render_size"])[:, 0:1, ...],
            inverted=data["ReverseZ"],
        ).squeeze()
        return depth_params

    def __init__(
        self,
        src_root: Path,
        dst_root: Path,
        seq_id: int,
        seq_path: Path,
        args: argparse.Namespace = None,
        scale: float = 2.0,
    ):
        # pylint: disable=duplicate-code
        super().__init__(
            src_root=src_root,
            dst_root=dst_root,
            seq_id=seq_id,
            seq_path=seq_path,
            args=args,
            version=type(self).__name__,
        )
        self.scale = scale
        # 2.0 -> 2, 1.5 -> 1_5, etc.
        self.scale_str = str(float(self.scale)).replace(".", "_").replace("_0", "")

        self._validate_exr_dataset_structure(
            self.src_root, self.seq_path, self.scale_str
        )

    def __iter__(self):
        # Dataset Binaries
        image_dirs = [
            f"x{self.scale_str}/depth",
            f"x{self.scale_str}/motion",
            f"x{self.scale_str}/color",
            "motion_gt",
            "ground_truth",
        ]

        self.seq_data = self.make_image_dict(image_dirs)

        self.max_frames = min(len(self.seq_data[k]) for k in self.seq_data)

        self.unique_seq_id = self.make_unique_seq_id(self.max_frames)

        # Create the iterator
        self.count = 0

        return self

    def __next__(self):
        if self.count >= self.max_frames:
            raise StopIteration

        # Feature to return
        out_features = {}

        # Read EXR data for current frame
        depth_raw = read_exr_torch(
            self.seq_data[f"x{self.scale_str}/depth"][self.count],
            dtype=np.float32,
            channels="R",
        )
        mv_lr_raw = read_exr_torch(
            self.seq_data[f"x{self.scale_str}/motion"][self.count],
            dtype=np.float16,
            channels="RG",
        )
        colour_raw = read_exr_torch(
            self.seq_data[f"x{self.scale_str}/color"][self.count],
            dtype=np.float16,
            channels="RGB",
        )
        mv_raw = read_exr_torch(
            self.seq_data["motion_gt"][self.count], dtype=np.float16, channels="RG"
        )
        truth_raw = read_exr_torch(
            self.seq_data["ground_truth"][self.count], dtype=np.float16, channels="RGB"
        )

        out_features["EmulatedFramerate"] = torch.tensor(
            self.metadata["EmulatedFramerate"], dtype=torch.float32
        ).reshape((1, 1))

        res_x, res_y = (
            self.metadata["TargetResolution"]["X"],
            self.metadata["TargetResolution"]["Y"],
        )
        out_features["TargetResolution"] = torch.tensor(
            [res_x, res_y], dtype=torch.int32
        ).reshape((1, 2))

        out_features["Samples"] = torch.tensor(
            self.metadata["Samples"]["Count"], dtype=torch.int32
        ).reshape((1, 1))

        frame_meta_data = self.metadata["Frames"][self.count]

        # Render sizes
        out_features["render_size"] = render_size = torch.tensor(
            [colour_raw.shape[2], colour_raw.shape[3]], dtype=torch.int32
        ).reshape((1, 2))
        out_features["outDims"] = torch.tensor(
            [truth_raw.shape[2], truth_raw.shape[3]], dtype=torch.int32
        ).reshape((1, 2))

        # View Projection matrix is correct, no need for transposition (fixed after 24_10)
        out_features["viewProj"] = torch.reshape(
            torch.tensor(frame_meta_data["ViewProjection"]), (1, 1, 4, 4)
        )

        # Transform motion vectors to PyTorch format
        def process_motion(motion: torch.tensor) -> torch.tensor:
            # clamp to [-1, 1]
            motion = torch.clamp(
                motion, -torch.ones_like(motion), torch.ones_like(motion)
            )
            # switch uv -> vu
            motion = swizzle(motion, "yx")
            # scale from uv to pixels
            size = torch.tensor(
                [motion.shape[2], motion.shape[3]], dtype=torch.float16
            ).reshape((1, 2, 1, 1))
            motion = motion * size
            return motion

        out_features["motion_lr"] = process_motion(mv_lr_raw)
        out_features["motion"] = process_motion(mv_raw)

        reverse_z = self.metadata["ReverseZ"]
        out_features["ReverseZ"] = torch.tensor(reverse_z, dtype=torch.bool).reshape(
            (1, 1)
        )
        if reverse_z:
            # Invert depth
            depth_raw = 1.0 - depth_raw.to(torch.float32)
            out_features["depth"] = depth_raw
        else:
            # Depth is usable as-is, **should not be inverted**
            out_features["depth"] = depth_raw.to(torch.float32)

        # Transform colour data
        out_features["colour_linear"] = colour_raw
        if not self.args.linear_truth:
            truth_raw = tonemap_inverse(truth_raw, mode="karis")
        out_features["ground_truth_linear"] = truth_raw

        # Depth planes
        out_features["infinite_zFar"] = torch.tensor(
            frame_meta_data["CameraFarPlane"] == -1, dtype=torch.bool
        ).reshape((1, 1))

        # This is done for backwards compatability sake, for when
        # we are confronted with certain game engines' infinite FarPlane.
        z_far = (
            5000.0
            if out_features["infinite_zFar"]
            else frame_meta_data["CameraFarPlane"]
        )
        out_features["zFar"] = torch.tensor(z_far, dtype=torch.float32).reshape((1, 1))
        out_features["zNear"] = torch.tensor(
            frame_meta_data["CameraNearPlane"], dtype=torch.float32
        ).reshape((1, 1))

        # Add additional MetaData features
        out_features["seq"] = torch.tensor(
            self.unique_seq_id, dtype=torch.int64
        ).reshape((1, 1))
        out_features["img"] = torch.tensor(self.count, dtype=torch.int64).reshape(
            (1, 1)
        )
        out_features["FovX"] = torch.tensor(
            frame_meta_data["FovX"], dtype=torch.float32
        ).reshape((1, 1))
        out_features["FovY"] = torch.tensor(
            frame_meta_data["FovY"], dtype=torch.float32
        ).reshape((1, 1))
        out_features["scale"] = torch.tensor(self.scale, dtype=torch.float32).reshape(
            (1, 1)
        )
        # Camera cuts reset recurrent history; default to False for legacy captures.
        camera_cut = bool(frame_meta_data.get("CameraCut", False))
        out_features["camera_cut"] = torch.tensor(camera_cut, dtype=torch.bool).reshape(
            (1, 1)
        )

        # Exposure is an optional field, so we must ensure it exists before using it
        if "Exposure" in frame_meta_data.keys():
            # Exposure output from Engine is assumed to already be exponential.
            # We take a natural log here, so this plays nicely with our
            # codebase, which will do exp(exposure)
            exposure_log = torch.log(torch.tensor(float(frame_meta_data["Exposure"])))
            out_features["exposure"] = exposure_log.to(torch.float32).reshape((1, 1))

        # Convert scale factor to relevant index
        scale_idx = self.metadata["UpscalingRatiosIndices"][f"x{self.scale_str}_index"]

        # Jitter Offsets
        x, y = (
            frame_meta_data["NormalizedPerRatioJitter"][scale_idx]["X"],
            frame_meta_data["NormalizedPerRatioJitter"][scale_idx]["Y"],
        )
        jitter = torch.tensor([y, x], dtype=torch.float32).reshape((1, 2, 1, 1))
        out_features["jitter"] = jitter * render_size.reshape((1, 2, 1, 1))

        # Historic Dataset Correction
        # Originally datasets were captured from game engines
        # which did not correct for display space Y-orientation
        # Game engines we use now correct for this, so we align by undoing this.
        out_features["Y"] = torch.tensor(-y, dtype=torch.float32).reshape((1, 1))
        out_features["X"] = torch.tensor(x, dtype=torch.float32).reshape((1, 1))

        # Depth params for FSR-style disocclusion mask
        out_features["depth_params"] = (
            self._create_depth_params(out_features).to(torch.float32).reshape((1, 4))
        )

        # Output data
        out_filepath = f"{Path(self.seq_path)}.safetensors"

        # Update Count
        self.count += 1

        return [(out_filepath, out_features)]
