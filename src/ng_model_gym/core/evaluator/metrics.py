# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from torchmetrics import Metric
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from ng_model_gym.core.utils.config_model import ConfigModel

logger = logging.getLogger(__name__)


def calculate_psnr(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate peak signal to noise ratio between predictions and targets.

    It is expected that the inputs to this are 2 tensors of 5 dimensions:
    - preds: Predictions from the model of shape (N,T,C,H,W)
    - target: Ground truth values of shape (N,T,C,H,W)
    """
    num_dims = len(preds.shape)  # will be 5D during training and evaluation.
    dims_to_reduce_over = tuple(range(2, num_dims))

    psnr = peak_signal_noise_ratio(
        preds,
        targets,
        data_range=1.0,
        reduction="elementwise_mean",
        dim=dims_to_reduce_over,
    )

    return psnr


def calculate_tpsnr(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate temporal peak signal to noise ratio between predictions and targets.

    It is expected that the inputs to this are 2 tensors of 5 dimensions:
    - preds: Predictions from the model of shape (N,T,C,H,W)
    - target: Ground truth values of shape (N,T,C,H,W)
    """
    num_dims = len(preds.shape)  # will be 5D during training and evaluation.
    dims_to_reduce_over = tuple(range(2, num_dims))

    tpsnr_pred = preds[:, 1:, ...] - preds[:, :-1, ...]
    tpsnr_target = targets[:, 1:, ...] - targets[:, :-1, ...]
    tpsnr = peak_signal_noise_ratio(
        tpsnr_pred,
        tpsnr_target,
        data_range=1.0,
        reduction="elementwise_mean",
        dim=dims_to_reduce_over,
    )

    return tpsnr


def calculate_recpsnr(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate recurrent peak signal to noise ratio between predictions and targets.

    It is expected that the inputs to this are 2 tensors of 5 dimensions:
    - preds: Predictions from the model of shape (N,T,C,H,W)
    - target: Ground truth values of shape (N,T,C,H,W)
    """
    num_dims = len(preds.shape)  # will be 5D during training and evaluation.
    dims_to_reduce_over = tuple(range(1, (num_dims - 1)))

    recpsnr_pred = preds[:, -1, ...]
    recpsnr_target = targets[:, -1, ...]
    recpsnr = peak_signal_noise_ratio(
        recpsnr_pred,
        recpsnr_target,
        data_range=1.0,
        reduction="elementwise_mean",
        dim=dims_to_reduce_over,
    )

    return recpsnr


def calculate_ssim(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate structural similarity index measure between predictions and targets.
     It is expected that the inputs to this are 2 tensors of 5 dimensions or 4 dimensions:
    - preds: Predictions from the model of shape (N,T,C,H,W) or (N,C,H,W)
    - target: Ground truth values of shape (N,T,C,H,W) or (N,C,H,W)
    """
    if preds.dim() == 5 or targets.dim() == 5:
        (
            N,
            T,
            C,
            H,
            W,
        ) = preds.shape  # will be 5D during training and evaluation, reshape to 4D
        preds = preds.view(N * T, C, H, W)
        targets = targets.view(N * T, C, H, W)

    # If 4D already, do nothing

    ssim = StructuralSimilarityIndexMeasure(
        data_range=1.0, kernel_size=11, sigma=1.5, gaussian_kernel=True
    ).to(preds.device)

    return ssim(preds, targets)


# pylint: disable=no-member
class Psnr(Metric):
    """PSNR

    As input to forward and update the metric accepts the following input

    - preds: Predictions from the model of shape (N,T,C,H,W)
    - target: Ground truth values of shape (N,T,C,H,W)
    """

    is_differentiable = False
    full_state_update = True
    is_streaming = False

    def __str__(self):
        return "PSNR"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_psnr", default=torch.tensor(0.0))
        self.add_state("total_steps", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update state variables"""
        if preds.shape != targets.shape:
            raise ValueError("preds and targets must have the same shape")

        psnr = calculate_psnr(preds, targets)

        if not torch.isinf(psnr):
            self.total_psnr += psnr
            self.total_steps += 1

    def compute(self) -> torch.Tensor:
        """Calculate PSNR"""
        return self.total_psnr / self.total_steps


class TPsnr(Metric):
    """Temporal PSNR, this is the PSNR for the delta between current and previous frames.

    As input to forward and update the metric accepts the following input

    - preds: Predictions from the model of shape (N,T,C,H,W)
    - target: Ground truth values of shape (N,T,C,H,W)
    """

    is_differentiable = False
    full_state_update = True
    is_streaming = False

    def __str__(self):
        return "tPSNR"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_tpsnr", default=torch.tensor(0.0))
        self.add_state("total_steps", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update state variables"""
        if preds.shape != targets.shape:
            raise ValueError("preds and targets must have the same shape")

        if preds.shape[1] < 2:
            raise ValueError(
                "preds and targets must have at least 2 frames for tPSNR calculation"
            )

        # Update states.
        tpsnr = calculate_tpsnr(preds, targets)

        if not torch.isinf(tpsnr):
            self.total_tpsnr += tpsnr
            self.total_steps += 1

    def compute(self) -> torch.Tensor:
        """Calculate temporal PSNR"""
        return self.total_tpsnr / self.total_steps


class RecPsnr(Metric):
    """Recurrent PSNR

    As input to forward and update the metric accepts the following input

    - preds: Predictions from the model of shape (N,T,C,H,W)
    - target: Ground truth values of shape (N,T,C,H,W)
    """

    is_differentiable = False
    full_state_update = True
    is_streaming = False

    def __str__(self):
        return "recPSNR"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_recpsnr", default=torch.tensor(0.0))
        self.add_state("total_steps", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update state variables, for sequences PSNR is only calculated on the last frame."""
        if preds.shape != targets.shape:
            raise ValueError("preds and targets must have the same shape")

        if preds.shape[1] < 2:
            raise ValueError(
                "preds and targets must have at least 2 frames for recPSNR calculation"
            )

        recpsnr = calculate_recpsnr(preds, targets)

        if not torch.isinf(recpsnr):
            self.total_recpsnr += recpsnr
            self.total_steps += 1

    def compute(self) -> torch.Tensor:
        """Calculate recurrent PSNR"""
        return self.total_recpsnr / self.total_steps


class Ssim(Metric):
    """SSIM metric

    As input to forward and update the metric accepts the following input

    - preds: Predictions from the model of shape (N,T,C,H,W)
    - target: Ground truth values of shape (N,T,C,H,W)
    """

    is_differentiable = False
    full_state_update = True
    is_streaming = False

    def __str__(self):
        return "SSIM"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_ssim", default=torch.tensor(0.0))
        self.add_state("total_steps", default=torch.tensor(0))

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update state variables."""
        if preds.shape != targets.shape:
            raise ValueError("preds and targets must have the same shape")

        ssim = calculate_ssim(preds, targets)

        self.total_ssim += ssim
        self.total_steps += 1

    def compute(self) -> torch.Tensor:
        """Calculate SSIM"""
        return self.total_ssim / self.total_steps


class TPsnrStreaming(Metric):
    """Temporal PSNR, this is the PSNR for the delta between current and previous frames.

    As input to forward and update the metric accepts the following input

    This version is intended for streaming evaluation where N=T=1, and we are manually feeding
    the sequence for evaluation.

    - preds: Predictions from the model of shape (N,T,C,H,W)
    - target: Ground truth values of shape (N,T,C,H,W)
    """

    is_differentiable = False
    full_state_update = True
    is_streaming = True

    def __str__(self):
        return "tPSNRStreaming"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_sequence_means", default=torch.tensor(0.0))
        self.add_state("total_sequences", default=torch.tensor(0))
        self.add_state("sequence_psnr_sum", default=torch.tensor(0.0))
        self.add_state("sequence_steps", default=torch.tensor(0))
        self.add_state("prev_pred", default=[])
        self.add_state("prev_target", default=[])
        self.add_state("prev_seq_id", default=torch.tensor(float("nan")))

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor, seq_id: torch.Tensor
    ) -> None:
        """Update state variables"""

        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        if preds.shape[0] > 1 or preds.shape[1] > 1:
            raise ValueError(
                """preds and targets must have only 1 frame and a batch size of 1
                to use TPsnrStreaming."""
            )

        if seq_id != self.prev_seq_id:  # pylint: disable=E0203
            if self.sequence_steps > 0:
                # Finish old sequence
                self.sum_sequence_means += self.sequence_psnr_sum / self.sequence_steps
                self.total_sequences += 1
            self.prev_pred.clear()
            self.prev_target.clear()
            self.prev_seq_id = seq_id
            self.sequence_psnr_sum *= 0
            self.sequence_steps *= 0

        # Update states.
        frame_pred, frame_target = preds[:, 0], targets[:, 0]
        if self.prev_pred:
            tpsnr_pred = frame_pred - self.prev_pred.pop(0)
            tpsnr_target = frame_target - self.prev_target.pop(0)

            tpsnr = calculate_psnr(tpsnr_pred.unsqueeze(1), tpsnr_target.unsqueeze(1))

            if not torch.isinf(tpsnr):
                self.sequence_psnr_sum += tpsnr
                self.sequence_steps += 1

        # Store for next step.
        self.prev_pred.append(frame_pred)
        self.prev_target.append(frame_target)

    def compute(self) -> torch.Tensor:
        """Calculate temporal PSNR"""
        total_means = self.sum_sequence_means
        num_seqs = self.total_sequences

        # Include the current in progress sequence.
        if self.sequence_steps > 0:
            total_means = total_means + (self.sequence_psnr_sum / self.sequence_steps)
            num_seqs = num_seqs + 1

        return total_means / num_seqs


class RecPsnrStreaming(Metric):
    """Recurrent PSNR, this is the PSNR for the last frame in a sequence.

    As input to forward and update the metric accepts the following input

    This version is intended for streaming evaluation where N=T=1, and we are manually feeding
    the sequence for evaluation.

    - preds: Predictions from the model of shape (N,T,C,H,W)
    - target: Ground truth values of shape (N,T,C,H,W)
    """

    is_differentiable = False
    full_state_update = True
    is_streaming = True

    def __str__(self):
        return "recPSNRStreaming"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_recpsnr", default=torch.tensor(0.0))
        self.add_state("total_steps", default=torch.tensor(0))
        self.add_state("prev_psnr", default=torch.tensor(0.0))
        self.add_state("prev_seq_id", default=torch.tensor(float("nan")))

    def update(
        self, preds: torch.Tensor, targets: torch.Tensor, seq_id: torch.Tensor
    ) -> None:
        """Update state variables, for sequences PSNR is only calculated on the last frame."""
        if preds.shape != targets.shape:
            raise ValueError("preds and target must have the same shape")

        if preds.shape[0] > 1 or preds.shape[1] > 1:
            raise ValueError(
                """preds and targets must have only 1 frame and a batch size of 1
                to use RecPsnrStreaming."""
            )

        if seq_id == self.prev_seq_id:  # pylint: disable=E0203
            self.total_recpsnr -= self.prev_psnr  # pylint: disable=E0203
            self.total_steps -= 1
        self.prev_seq_id = seq_id

        num_dims = len(preds.shape)  # will be 5D during training and evaluation.
        dims_to_reduce_over = tuple(range(1, (num_dims - 1)))

        # Update states.
        recpsnr_pred = preds[:, -1, ...]
        recpsnr_target = targets[:, -1, ...]
        recpsnr = peak_signal_noise_ratio(
            recpsnr_pred,
            recpsnr_target,
            data_range=1.0,
            reduction="elementwise_mean",
            dim=dims_to_reduce_over,
        )

        if not torch.isinf(recpsnr):
            self.total_recpsnr += recpsnr
            self.prev_psnr = recpsnr
        else:
            self.prev_psnr *= 0.0
        self.total_steps += 1

    def compute(self) -> torch.Tensor:
        """Calculate recurrent PSNR"""
        return self.total_recpsnr / self.total_steps


# pylint: enable=no-member


_METRIC_FACTORY: dict[str, type[Metric]] = {
    "PSNR": Psnr,
    "tPSNR": TPsnr,
    "RecPSNR": RecPsnr,
    "SSIM": Ssim,
    "tPSNRStreaming": TPsnrStreaming,
    "RecPSNRStreaming": RecPsnrStreaming,
}


# Mapping from non-streaming to streaming metric names
_STREAMING_METRIC_MAP: dict[str, str] = {
    "tPSNR": "tPSNRStreaming",
    "RecPSNR": "RecPSNRStreaming",
}


def _build_metric(metric_name: str) -> Metric:
    if metric_name not in _METRIC_FACTORY:
        raise KeyError(f"Unsupported metric '{metric_name}' in metrics configuration")
    metric_cls = _METRIC_FACTORY[metric_name]
    return metric_cls()


def _metrics_from_config(
    params: ConfigModel | None, use_streaming: bool
) -> list[Metric]:
    if params is None or params.metrics is None:
        return []

    metric_names = list(params.metrics)

    if use_streaming:
        metric_names = [_STREAMING_METRIC_MAP.get(name, name) for name in metric_names]

    return [_build_metric(name) for name in metric_names]


def _default_metrics(use_streaming: bool) -> list[Metric]:
    if use_streaming:
        return [Psnr(), TPsnrStreaming(), RecPsnrStreaming(), Ssim()]

    return [Psnr(), TPsnr(), RecPsnr(), Ssim()]


def get_metrics(
    params: ConfigModel | None = None,
    is_test: bool = False,
) -> list[Metric]:
    """Return list of metrics to use for evaluation.

    For evaluation/test, we use the streaming versions of temporal metrics.
    """
    use_streaming = is_test and params.dataset.recurrent_samples is not None
    metrics_from_config = _metrics_from_config(params, use_streaming)
    metrics = (
        metrics_from_config if metrics_from_config else _default_metrics(use_streaming)
    )
    metric_names = ", ".join(str(metric) for metric in metrics)
    logger.info("Tracking metrics %s", metric_names)
    return metrics
