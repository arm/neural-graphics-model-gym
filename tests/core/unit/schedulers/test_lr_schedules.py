# SPDX-FileCopyrightText: <text>Copyright 2024-2026 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0
import unittest

import torch
from pydantic import ValidationError

from ng_model_gym.core.schedulers import CosineAnnealingWithWarmupLR
from ng_model_gym.core.trainer import get_lr_schedule
from ng_model_gym.core.utils.config_model import ConfigModel
from ng_model_gym.core.utils.types import TrainEvalMode
from tests.testing_utils import create_simple_params, validate_params


class TestLRScheduleFunctions(unittest.TestCase):
    """Tests for LR schedule function in trainer.py."""

    def setUp(self):
        """Setup test case"""
        torch.manual_seed(0)
        self.params = create_simple_params()
        self.params.model_train_eval_mode = TrainEvalMode.FP32

    def test_cosine_annealing_schedule(self):
        """Test cosine annealing scheduler matches reference"""

        test_weights = [torch.ones([10])]
        lr = 0.003
        train_size = 75
        min_lr = 5e-5

        self.params.train.batch_size = 8
        self.params.train.fp32.number_of_epochs = 16
        self.params.train.fp32.lr_scheduler.min_lr = min_lr
        self.params.train.fp32.lr_scheduler.type = "cosine_annealing"

        optimizer = torch.optim.Adam(test_weights, lr=lr)

        lr_sched = get_lr_schedule(
            self.params.train.fp32,
            optimizer,
            train_size // self.params.train.batch_size,
        )
        self.assertIsInstance(lr_sched, CosineAnnealingWithWarmupLR)

        torch_lr_produced = []
        for _ in range(train_size):
            if lr_sched.get_last_lr()[0] not in torch_lr_produced:
                torch_lr_produced.append(lr_sched.get_last_lr()[0])
            lr_sched.step()

        # Golden cosine annealing scheduler values - length is 34
        golden = [
            5e-05,
            0.0004714285714285714,
            0.0008928571428571428,
            0.001314285714285714,
            0.0017357142857142855,
            0.0021571428571428575,
            0.0025785714285714283,
            0.003,
            0.0029996122055362968,
            0.0029984490260564337,
            0.002996511073186939,
            0.0029937993659480055,
            0.002990315330217674,
            0.0029860607979820698,
            0.002981038006372103,
            0.002975249596487136,
            0.002968698612006228,
            0.0029613884975877025,
            0.0029533230970578655,
            0.002944506651389833,
            0.00293494379647353,
            0.002924639560678033,
            0.0029135993622075368,
            0.00290182900625234,
            0.002889334681936341,
            0.002876122959062656,
            0.002862200784659062,
            0.002847575479325094,
            0.0028322547333827024,
            0.0028162466028325054,
            0.0027995595051177593,
            0.002782202214698274,
            0.0027641838584365973,
            0.0027455139107989035,
        ]

        for i, golden_val in enumerate(golden):
            with self.subTest(index=i):
                self.assertAlmostEqual(
                    golden_val,
                    torch_lr_produced[i],
                    places=7,
                    msg=f"Learning rate different at index {i}",
                )

    def test_exponential_step_schedule(self):
        """Test exponential step learning rate schedule decreases every step_size."""

        test_weights = [torch.ones([10])]
        lr = 0.01
        train_size = 5
        epochs = 10

        params_dict = self.params.model_dump()
        params_dict["train"]["fp32"]["lr_scheduler"] = {}  # Remove default lr_scheduler

        params_dict["train"]["fp32"]["number_of_epochs"] = epochs
        params_dict["train"]["fp32"]["lr_scheduler"]["type"] = "exponential_step"
        params_dict["train"]["fp32"]["lr_scheduler"]["decay_rate"] = 0.977
        params_dict["train"]["fp32"]["lr_scheduler"]["decay_factor"] = 5
        self.params = validate_params(params_dict)

        decay_step_size = (epochs * train_size) // 5

        optimizer = torch.optim.Adam(test_weights, lr=lr)
        lr_sched = get_lr_schedule(self.params.train.fp32, optimizer, train_size)

        self.assertIsInstance(lr_sched, torch.optim.lr_scheduler.StepLR)
        self.assertEqual(lr_sched.gamma, self.params.train.fp32.lr_scheduler.decay_rate)
        self.assertEqual(lr_sched.step_size, decay_step_size)

        old_lr = lr_sched.get_last_lr()[0]
        for _ in range(train_size * epochs):
            lr_sched.step()
            if (lr_sched.last_epoch % decay_step_size) == 0:
                new_lr = lr_sched.get_last_lr()[0]
                self.assertAlmostEqual(new_lr, old_lr * lr_sched.gamma, places=7)
                old_lr = new_lr

    def test_static_schedule(self):
        """Test static learning rate schedule returns None."""

        test_weights = [torch.ones([10])]
        lr = 0.01
        train_size = 20000
        epochs = 15

        self.params.train.fp32.number_of_epochs = epochs
        self.params.train.fp32.lr_scheduler.type = "static"
        optimizer = torch.optim.Adam(test_weights, lr=lr)

        lr_sched = get_lr_schedule(self.params.train.fp32, optimizer, train_size)

        self.assertIsNone(lr_sched)

    def test_unknown_schedule(self):
        """Test unknown learning rate raises an error."""

        test_weights = [torch.ones([10])]
        lr = 0.01
        train_size = 20000
        epochs = 15

        self.params.train.fp32.number_of_epochs = epochs
        self.params.train.fp32.lr_scheduler.type = "abcd"
        optimizer = torch.optim.Adam(test_weights, lr=lr)

        with self.assertRaises(ValueError):
            get_lr_schedule(self.params.train.fp32, optimizer, train_size)


class TestSchedulerConfig(unittest.TestCase):
    """Tests for discriminated union LR scheduler configuration."""

    # pylint: disable=C0116
    def setUp(self):
        """Run before each test case"""
        self.params = create_simple_params().model_dump()

    def test_cosine_annealing_valid(self):
        params = self.params
        params["train"]["fp32"]["lr_scheduler"] = {
            "type": "cosine_annealing",
            "warmup_percentage": 0.1,
            "min_lr": 5e-5,
        }
        cfg = ConfigModel(**params)
        self.assertEqual(cfg.train.fp32.lr_scheduler.type, "cosine_annealing")
        self.assertEqual(cfg.train.fp32.lr_scheduler.min_lr, 5e-5)

    def test_exponential_valid(self):
        params = self.params
        params["train"]["qat"]["lr_scheduler"] = {
            "type": "exponential_step",
            "decay_rate": 0.9,
            "decay_factor": 5,
        }
        cfg = ConfigModel(**params)
        self.assertEqual(cfg.train.qat.lr_scheduler.type, "exponential_step")
        self.assertEqual(cfg.train.qat.lr_scheduler.decay_factor, 5)

    def test_static_valid(self):
        params = self.params
        params["train"]["fp32"]["lr_scheduler"] = {"type": "static"}
        cfg = ConfigModel(**params)
        self.assertEqual(cfg.train.fp32.lr_scheduler.type, "static")

    def test_missing_type_fails(self):
        params = self.params
        params["train"]["fp32"]["lr_scheduler"] = {
            "warmup_percentage": 0.1,
            "min_lr": 1e-4,
        }
        with self.assertRaises(ValidationError):
            ConfigModel(**params)

    def test_static_with_extra_field_fails(self):
        params = self.params
        params["train"]["fp32"]["lr_scheduler"] = {
            "type": "static",
            "min_lr": 1e-4,  # illegal for static
        }
        with self.assertRaises(ValidationError):
            ConfigModel(**params)

    def test_exponential_step_missing_required_field(self):
        params = self.params
        params["train"]["fp32"]["lr_scheduler"] = {
            "type": "exponential_step",
            "decay_rate": 0.95,
            # decay_factor missing
        }
        with self.assertRaises(ValidationError):
            ConfigModel(**params)

    def test_wrong_field_for_exponential_step(self):
        params = self.params
        params["train"]["fp32"]["lr_scheduler"] = {
            "type": "exponential_step",
            "warmup_percentage": 0.1,  # illegal for exponential_step
            "decay_factor": 0.9,
            "decay_step": 10,
        }
        with self.assertRaises(ValidationError):
            ConfigModel(**params)

    # pylint: enable=C0116
