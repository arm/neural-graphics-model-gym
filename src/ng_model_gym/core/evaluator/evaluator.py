# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torchvision
from tqdm.auto import tqdm

from ng_model_gym.core.data.dataloader import get_dataloader
from ng_model_gym.core.data.utils import DataLoaderMode, move_to_device
from ng_model_gym.core.evaluator.metrics import get_metrics
from ng_model_gym.core.model.base_ng_model_wrapper import BaseNGModelWrapper
from ng_model_gym.core.model.model_factory import BaseNGModel
from ng_model_gym.core.model.recurrent_model import FeedbackModel
from ng_model_gym.core.utils.general_utils import create_directory

logger = logging.getLogger(__name__)


class NGModelEvaluator:
    """This class is used to evaluate a model end to end"""

    def __init__(self, model: BaseNGModel | BaseNGModelWrapper, params):
        self.model = model
        self.params = params
        self.out_dir = self.params.output.dir
        self.export_png_dir = (
            Path(self.out_dir, "png") if self.params.output.export_frame_png else None
        )
        self.metrics = get_metrics(self.params, is_test=True)
        self.dataloader = None
        self.idx = 0
        self.x_in = None
        self.y_true = None
        self.y_pred = None
        self.results = {}

        for metric in self.metrics:
            metric.to(self.model.device)
            self.results[str(metric)] = {}

        if isinstance(model, FeedbackModel):
            # Set temporary values for parameters to allow evaluation of test set
            logger.debug("Temporarily setting recurrent_samples to 1")
            model.recurrent_samples = 1

    def prepare_datasets(self):
        """Load test dataset for evaluation"""

        self.dataloader = get_dataloader(
            self.params,
            num_workers=self.params.dataset.num_workers,
            prefetch_factor=self.params.dataset.prefetch_factor,
            loader_mode=DataLoaderMode.TEST,
        )

    def evaluate(self, profiler: Optional[torch.profiler.profile] = None):
        """Do evaluation."""

        # Initialise start of evaluation.
        self._test_begin()

        evaluate_pbar = tqdm(
            enumerate(self.dataloader, 0),
            total=len(self.dataloader),
            desc="Evaluation",
            leave=True,
        )

        # Run evaluation.
        for self.idx, (self.x_in, self.y_true) in evaluate_pbar:
            # Ensure input data and ground truth are on the same device as the model.
            self.x_in = move_to_device(self.x_in, self.model.device)
            self.y_true = move_to_device(self.y_true, self.model.device)

            # Run Inference on model - gradients not needed as we only evaluate.
            with torch.no_grad():
                self._run_model()

            # Predict End
            self._predict_end()

            if profiler:
                profiler.step()

            # Update progress bar
            self._update_progress_bar(evaluate_pbar)

        # End of Test
        self._test_end()

        self._save_results_json()

    def _run_model(self):
        """Invoke single forward pass."""
        self.y_pred = self.model(self.x_in)["output"]

    def _update_progress_bar(self, pbar, update_interval=1):
        if self.idx % update_interval == 0:
            metric_string = self._get_results_string()
            mode = "Evaluation"
            pbar.set_description(f"{mode}: {metric_string}")

    def _get_results_string(self):
        """Returns the current results as a string.

        This will be the running average and not the metric value on the current batch.
        """
        results = self.get_results()

        metric_string = " ".join(
            [f"{key}: {value:.4f}, " for (key, value) in results.items()]
        )
        return metric_string

    def get_results(self):
        """Return dictionary with results, update results table.

        This will be the running average and not the metric value on the current batch.
        """
        results = {}
        if self.metrics is not None:
            for metric in self.metrics:
                result = metric.compute()
                results[str(metric)] = result
                self.results[str(metric)][str(self.idx)] = result
        return results

    def _test_begin(self):
        """Called before evaluation begins."""
        self.prepare_datasets()
        create_directory(self.out_dir)
        if self.export_png_dir:
            logger.warning(
                "Exporting .png frames is selected, this may slow down evaluation."
            )
            create_directory(self.export_png_dir / "predicted")
            create_directory(self.export_png_dir / "ground_truth")
        self.model.eval()
        if isinstance(self.model, FeedbackModel):
            self.model.reset_history_buffers()

    def _test_end(self):
        """Called after whole dataset has been evaluated."""
        metric_string = self._get_results_string()
        logger.info("-------------- Evaluation Complete --------------")
        logger.info(metric_string)
        if not Path(self.out_dir).exists():
            create_directory(self.out_dir)
        full_path = Path(self.out_dir, "results.log")
        with open(full_path, "w", encoding="utf-8") as results_file:
            results_file.write(metric_string)

    def _predict_end(self):
        """Called after single batch has evaluated."""
        if self.metrics is not None:
            for metric in self.metrics:
                if metric.is_streaming:
                    # Streaming metrics need to know when we start a new sequence.
                    metric.update(self.y_pred, self.y_true, seq_id=self.x_in["seq"])
                else:
                    metric.update(self.y_pred, self.y_true)
        if self.export_png_dir:
            torchvision.utils.save_image(
                self.y_pred[0],
                self.export_png_dir / "predicted" / f"frame_{self.idx:04d}_pred.png",
            )
            torchvision.utils.save_image(
                self.y_true[0],
                self.export_png_dir / "ground_truth" / f"frame_{self.idx:04d}_gt.png",
            )

    def _save_results_json(self):
        to_json = {}
        for metric in self.metrics:
            to_json[str(metric)] = {
                k: None if math.isnan(v.item()) else v.item()
                for k, v in self.results[str(metric)].items()
            }

        time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filepath_results = Path(self.out_dir, f"eval_metrics_{time_stamp}.json")
        with open(filepath_results, "w", encoding="utf-8") as json_file:
            json.dump(to_json, json_file, indent=4)
            logger.info(f"Saved evaluation metrics: {filepath_results}")
