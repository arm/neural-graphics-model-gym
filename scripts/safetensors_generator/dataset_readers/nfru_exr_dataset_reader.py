# SPDX-FileCopyrightText: <text>Copyright 2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from ng_model_gym.core.data import tonemap_forward
from scripts.safetensors_generator.dataset_readers.safetensors_feature_iterator import (
    EXRDatasetReader,
)
from scripts.safetensors_generator.exr_utils import read_exr_torch
from scripts.safetensors_generator.fsr2_methods import depth_to_view_space_params


class NFRUEXRDatasetReader(EXRDatasetReader):
    """Dataset reader for NFRU and EXR files v2.2 spec."""

    @staticmethod
    def _create_depth_params(data: Dict[str, torch.Tensor]) -> torch.Tensor:
        MakeImageLike = lambda t: t.unsqueeze(-1).unsqueeze(-1)
        depth_params = depth_to_view_space_params(
            zNear=MakeImageLike(data["NearPlane"].reshape((1, 1))),
            zFar=MakeImageLike(data["FarPlane"].reshape((1, 1))),
            FovY=MakeImageLike(data["FovY"].reshape((1, 1))),
            infinite=MakeImageLike(data["infinite_zFar"].reshape((1, 1))).to(
                torch.bool
            ),
            renderSizeWidth=MakeImageLike(data["render_size"].reshape((1, 2)))[
                :, 1:2, ...
            ]
            / 2.0,
            renderSizeHeight=MakeImageLike(data["render_size"].reshape((1, 2)))[
                :, 0:1, ...
            ]
            / 2.0,
            inverted=torch.tensor(False, dtype=torch.bool),
        ).squeeze()
        return depth_params

    @staticmethod
    def _calculate_motion(
        depth: torch.Tensor,
        viewProj: torch.Tensor,
        viewProj_prev: torch.Tensor,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        # Helper Methods
        Float = lambda t: torch.tensor(t, dtype=torch.float32)

        def swizzle_xy_invY(t: torch.Tensor) -> torch.Tensor:
            x, y = torch.split(t, split_size_or_sections=1, dim=1)
            scale = -1.0
            return torch.concat([y * scale, x], dim=1)

        def calc_uv(frame: torch.Tensor) -> torch.Tensor:
            sh = frame.shape
            height, width = float(sh[2]), float(sh[3])

            jj, ii = torch.meshgrid(
                torch.arange(0, height, device=frame.device, dtype=torch.float32),
                torch.arange(0, width, device=frame.device, dtype=torch.float32),
                indexing="ij",
            )
            # need to clone due to multiprocessing memory location issues
            jj = jj.clone()
            ii = ii.clone()

            jj /= height
            jj += 0.5 / height
            jj = 1.0 - jj

            ii /= width
            ii += 0.5 / width

            uv = torch.stack([ii, jj], dim=0).unsqueeze(0)
            return uv

        # Calculate dimensions
        depth = depth.to(torch.float32)
        sh = depth.shape
        height, width = sh[2], sh[3]
        outDims = torch.reshape(
            torch.stack([Float(width), Float(height)], dim=-1), (1, 2, 1, 1)
        )

        # Convert from: clip-space-current -> world-space -> clip-space-previous
        motionMat = viewProj_prev @ torch.linalg.inv(viewProj)
        motionMat = torch.reshape(motionMat, (1, 1, 1, 4, 4))

        # Caclulate screen-space UV coordinates
        uvT = calc_uv(depth)

        # Clip-Space Position
        clip = torch.concat([2.0 * uvT - 1.0, depth, torch.ones_like(depth)], dim=1)
        clip = clip.unsqueeze(-1)

        # Reprojection and normalization
        torch.permute(clip, [0, 2, 3, 1, 4])
        # reproj = motionMat @ clip
        reproj = torch.permute(
            torch.squeeze(motionMat @ torch.permute(clip, [0, 2, 3, 1, 4]), dim=-1),
            [0, 3, 1, 2],
        )
        reproj_xy, _, reproj_w = torch.split(
            reproj, split_size_or_sections=[2, 1, 1], dim=1
        )
        uvTm2 = ((reproj_xy / reproj_w) + 1.0) * 0.5

        # Swizzle x and y, negate y
        velocity = swizzle_xy_invY(((uvT - uvTm2) * outDims))
        velocity = velocity.to(dtype)

        # Remove NaN's, Inf's and filter small values when they occur
        basically_0_motion = 1e-3
        velocity = torch.where(torch.isinf(velocity), 0.0, velocity)
        velocity = torch.where(torch.isnan(velocity), 0.0, velocity)
        velocity = torch.where(abs(velocity) < basically_0_motion, 0.0, velocity)

        return velocity

    def __init__(
        self,
        src_root: Path,
        dst_root: Path,
        seq_id: int,
        seq_path: Path,
        args: argparse.Namespace,
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
        # Assume fixed scaling of 2.0
        self.scale_str = "x2"

    def __iter__(self):
        # Dataset Binaries
        image_dirs = [
            f"{self.scale_str}/depth",
            f"{self.scale_str}/motion_m2",
            f"{self.scale_str}/motion_m1",
            "ground_truth",
        ]

        self.seq_data = self.make_image_dict(image_dirs)

        self.max_frames = max(len(self.seq_data[k]) for k in self.seq_data)

        self.exposure = torch.math.exp(2.0)

        self.inverseY = False
        if "InverseY" in self.metadata:
            self.inverseY = self.metadata["InverseY"]

        # Globals
        self.dtype = torch.float16

        self.unique_seq_id = self.make_unique_seq_id(self.max_frames)

        # Create the iterator
        self.count = 0

        return self

    def _read_depth(self, count):
        depth = read_exr_torch(
            self.seq_data["x2/depth"][count], dtype=np.float32, channels="R"
        )
        return depth

    def _read_view_proj(self, tensor):
        # View Projection matrix is correct, no need for transposition (fixed after 24_10)
        viewProj = torch.reshape(torch.tensor(tensor), (4, 4))
        # Fix for flipped y data (e.g. in output from some game engines)
        if self.inverseY:
            # fmt:off
            inv_mat = torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0,-1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ],
            dtype=torch.float32)
            # fmt:on
            viewProj = (viewProj.T @ inv_mat).T

        return viewProj

    def _calculate_synthetic_mvs(self, out_features):
        mv = out_features["mv_{}_f30_m1"]
        depth = out_features["depth"]
        out_features = {}
        # NOTE: Frame-rate agnostic naming convention
        sy_t_f30_m1 = torch.zeros_like(mv, dtype=self.dtype)
        sy_t_f30_p1 = torch.zeros_like(mv, dtype=self.dtype)
        sy_t_f60_m1 = torch.zeros_like(mv, dtype=self.dtype)
        sy_t_f60_p1 = torch.zeros_like(mv, dtype=self.dtype)

        view_proj = self._read_view_proj(
            self.metadata["Frames"][self.count]["ViewProjection"]
        )
        # High frequency
        if self.count > 0 and self.count < self.max_frames - 1:
            view_proj_m1 = self._read_view_proj(
                self.metadata["Frames"][self.count - 1]["ViewProjection"]
            )
            view_proj_p1 = self._read_view_proj(
                self.metadata["Frames"][self.count + 1]["ViewProjection"]
            )
            sy_t_f60_m1 = self._calculate_motion(1.0 - depth, view_proj, view_proj_m1)
            sy_t_f60_p1 = self._calculate_motion(1.0 - depth, view_proj, view_proj_p1)

        # Low frequency
        if self.count > 1 and self.count < self.max_frames - 2:
            view_proj_m2 = self._read_view_proj(
                self.metadata["Frames"][self.count - 2]["ViewProjection"]
            )
            view_proj_p2 = self._read_view_proj(
                self.metadata["Frames"][self.count + 2]["ViewProjection"]
            )
            sy_t_f30_m1 = self._calculate_motion(1.0 - depth, view_proj, view_proj_m2)
            sy_t_f30_p1 = self._calculate_motion(1.0 - depth, view_proj, view_proj_p2)

        # NOTE: we inverse all motion here to align with popular conventions from game engines.
        out_features["sy_{}_f30_m1"] = -sy_t_f30_m1
        out_features["sy_{}_f30_p1"] = -sy_t_f30_p1
        out_features["sy_{}_f60_m1"] = -sy_t_f60_m1
        out_features["sy_{}_f60_p1"] = -sy_t_f60_p1
        out_features["ViewProj"] = view_proj
        return out_features

    def _read_rgb(self, count):
        rgb_linear = read_exr_torch(
            self.seq_data["ground_truth"][count], dtype=np.float32, channels="RGB"
        )
        rgb_reinhard = tonemap_forward(rgb_linear * self.exposure, mode="reinhard").to(
            dtype=self.dtype
        )
        return rgb_linear, rgb_reinhard

    def _read_lf_mv(self, count):
        if count % 2 != 0:
            # Return placeholder values for odd frames with no mvs
            return torch.zeros(1, 4, 540, 960)
        return read_exr_torch(
            self.seq_data["x2/motion_m2"][count // 2], dtype=np.float32, channels="RGBA"
        )

    def __next__(self) -> List[Tuple[Path, Dict]]:
        if self.count >= self.max_frames:
            raise StopIteration

        # Feature to return
        out_features = {}
        out_filepath = Path(self.seq_path).with_suffix(".safetensors")

        out_features["rgb_linear"], out_features["rgb_reinhard"] = self._read_rgb(
            self.count
        )
        nfru_shape = (
            out_features["rgb_linear"].shape[2],
            out_features["rgb_linear"].shape[3],
        )

        out_features["render_size"] = render_size = torch.tensor(
            nfru_shape, dtype=torch.int32
        )

        frame_meta_data = self.metadata["Frames"][self.count]

        # Exposure output from Engine is assumed to already be exponential. We
        # take a natural log here, so this plays nicely with our codebase,
        # which will do exp(exposure)
        exposure = torch.tensor(float(frame_meta_data["Exposure"]))
        out_features["exposure"] = torch.log(exposure).to(torch.float32).reshape(1)

        # Read EXR Data for current frame
        depth = self._read_depth(self.count)
        mv_nfru_raw = self._read_lf_mv(self.count)
        mv_nfru_raw_hf = read_exr_torch(
            self.seq_data["x2/motion_m1"][self.count], dtype=np.float32, channels="RGBA"
        )

        out_features["depth"] = depth

        viewProj = self._read_view_proj(
            self.metadata["Frames"][self.count]["ViewProjection"]
        )

        # MV: (540p) Velocity texture containing motion information from the frame's velocity
        # texture and the Ground Truth resolved velocity (GT Resolved velocity is calculated
        # choosing the velocity from the nearest pixel within the 8x8 tile of the 8k src texture).
        # We invert motion to align with popular uv conventions in game engines
        def unpack_and_scale_flow(t):
            mv, _ = torch.split(t, 2, dim=1)
            u, v = torch.split(mv, 1, dim=1)
            # NOTE: We invert mv direction here!
            u *= -u.shape[3]
            v *= -v.shape[2]
            return torch.concat([v, u], dim=1).to(self.dtype)

        out_features["mv_{}_f30_m1"] = unpack_and_scale_flow(mv_nfru_raw)
        out_features["mv_{}_f60_m1"] = unpack_and_scale_flow(mv_nfru_raw_hf)

        out_features |= self._calculate_synthetic_mvs(out_features)

        # Add additional MetaData features

        # Depth planes
        try:
            out_features["infinite_zFar"] = torch.tensor(
                self.metadata["FarPlane"] == -1, dtype=torch.bool
            ).reshape(1)
            farPlane = (
                5000.0 if out_features["infinite_zFar"] else self.metadata["FarPlane"]
            )
            out_features["FarPlane"] = torch.tensor(
                farPlane, dtype=torch.float32
            ).reshape(1)
            out_features["NearPlane"] = torch.tensor(
                self.metadata["NearPlane"], dtype=torch.float32
            ).reshape(1)
        except:  # pylint: disable=bare-except
            out_features["infinite_zFar"] = torch.tensor(
                frame_meta_data["CameraFarPlane"] == -1, dtype=torch.bool
            ).reshape(1)
            farPlane = (
                5000.0
                if out_features["infinite_zFar"]
                else frame_meta_data["CameraFarPlane"]
            )
            out_features["FarPlane"] = torch.tensor(
                farPlane, dtype=torch.float32
            ).reshape(1)
            out_features["NearPlane"] = torch.tensor(
                frame_meta_data["CameraNearPlane"], dtype=torch.float32
            ).reshape(1)

        out_features["ViewProj"] = viewProj.unsqueeze(0)
        out_features["seq_id"] = torch.tensor(
            self.unique_seq_id, dtype=torch.int64
        ).reshape(1)
        out_features["img_id"] = torch.tensor(self.count, dtype=torch.int64).reshape(1)
        out_features["FovX"] = torch.tensor(
            frame_meta_data["FovX"], dtype=torch.float32
        ).reshape(1)
        out_features["FovY"] = torch.tensor(
            frame_meta_data["FovY"], dtype=torch.float32
        ).reshape(1)
        out_features["InverseY"] = torch.tensor(
            self.inverseY, dtype=torch.int64
        ).reshape(1)

        # Depth params for FSR-style disocclusion mask
        out_features["DepthParams"] = (
            self._create_depth_params(out_features).to(torch.float32).reshape((1, 4))
        )

        # Convert scale factor to relevant index
        scale_idx = self.metadata["UpscalingRatiosIndices"][f"{self.scale_str}_index"]

        # pylint: disable=duplicate-code
        # Jitter Offsets
        x, y = (
            frame_meta_data["NormalizedPerRatioJitter"][scale_idx]["X"],
            frame_meta_data["NormalizedPerRatioJitter"][scale_idx]["Y"],
        )
        jitter = torch.tensor([y, x], dtype=torch.float32).reshape((1, 2, 1, 1))
        out_features["jitter"] = jitter * render_size.reshape((1, 2, 1, 1))
        # pylint: enable=duplicate-code

        self.count += 1

        return [(out_filepath, out_features)]
