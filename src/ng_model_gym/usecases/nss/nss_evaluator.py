# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
from ng_model_gym.core.evaluator.evaluator import BaseModelEvaluator
from ng_model_gym.usecases.nss.model.recurrent_model import FeedbackModel


class ModelEvaluator(BaseModelEvaluator):
    """Evaluates a trained model end-to-end"""

    def __init__(self, model: FeedbackModel, params):
        model.recurrent_samples = params.dataset.recurrent_samples
        super().__init__(model, params)

    def _run_model(self):
        """Invoke single forward pass."""
        self.y_pred = self.model(self.x_in)["output"]
