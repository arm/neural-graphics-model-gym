# SPDX-FileCopyrightText: <text>Copyright 2024-2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import expectedFailure

import numpy as np
import torch

from ng_model_gym.core.schedulers.lr_scheduler import CosineAnnealingWithWarmupLR
from ng_model_gym.core.trainer.trainer import get_lr_schedule
from ng_model_gym.core.utils.types import TrainEvalMode
from tests.testing_utils import create_simple_params


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
        self.params.train.fp32.cosine_annealing_scheduler_config.min_lr = min_lr

        optimizer = torch.optim.Adam(test_weights, lr=lr)

        lr_sched = get_lr_schedule(
            self.params.train.fp32, optimizer, train_size, self.params
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

    @expectedFailure
    def test_exponential_schedule(self):
        """Test exponential learning rate schedule matches tf.keras."""

        test_weights = [torch.ones([10])]
        lr = 0.01
        train_size = 20000
        epochs = 15

        self.params.train.fp32.number_of_epochs = epochs
        self.params.optimizer.exponential_scheduler_config.decay = 0.977
        self.params.optimizer.learning_rate_scheduler = "exponential"

        optimizer = torch.optim.Adam(test_weights, lr=lr)

        lr_sched = get_lr_schedule(
            self.params.train.fp32, optimizer, train_size, self.params
        )

        torch_lr_produced = []
        for _ in range(epochs):
            for _ in range(train_size):
                if lr_sched.get_last_lr()[0] not in torch_lr_produced:
                    torch_lr_produced.append(lr_sched.get_last_lr()[0])
                lr_sched.step()

        tf_keras_lr_schedule_values = [
            0.009999999776482582,
            0.009769999422132969,
            0.009545289911329746,
            0.009325748309493065,
            0.009111255407333374,
            0.00890169758349657,
            0.008696957491338253,
            0.00849692802876234,
            0.008301498368382454,
            0.008110564202070236,
            0.00792402122169733,
            0.007741768844425678,
            0.007563707884401083,
            0.007389742415398359,
            0.007219778839498758,
            0.007053723558783531,
            0.006891487631946802,
            0.006732983514666557,
            0.006578125059604645,
            0.006426828447729349,
            0.006279011256992817,
            0.0061345938593149185,
            0.005993498023599386,
            0.005855647847056389,
            0.005720967426896095,
            0.005589385516941547,
            0.005460829474031925,
            0.0053352308459579945,
            0.005212520249187946,
            0.00509263202548027,
            0.004975501913577318,
            0.004861065186560154,
            0.00474926084280014,
            0.004640027415007353,
            0.004533306695520878,
            0.004429040942341089,
            0.004327172879129648,
            0.00422764802351594,
            0.004130411893129349,
            0.004035412799566984,
            0.00394259812310338,
            0.0038519182708114386,
            0.003763324348255992,
            0.0036767679266631603,
            0.0035922019742429256,
            0.003509581321850419,
            0.0034288610331714153,
            0.0033499973360449076,
            0.0032729473896324635,
            0.003197669517248869,
            0.0031241232063621283,
            0.0030522681772708893,
            0.002982066012918949,
            0.0029134785290807486,
            0.0028464687056839466,
            0.002780999755486846,
            0.0027170367538928986,
            0.0026545450091362,
            0.0025934905279427767,
            0.00253384024836123,
            0.0024755618069320917,
            0.002418623771518469,
            0.002362995408475399,
            0.002308646449819207,
            0.0022555477917194366,
            0.002203670097514987,
            0.0021529856603592634,
            0.0021034670062363148,
            0.002055087126791477,
            0.002007820177823305,
            0.001961640315130353,
            0.0019165226258337498,
            0.0018724426627159119,
            0.001829376444220543,
            0.0017873008036985993,
            0.0017461928073316813,
            0.0017060304526239634,
            0.0016667917370796204,
            0.0016284554731100798,
            0.0015910009387880564,
            0.0015544079942628741,
            0.0015186566160991788,
            0.001483727479353547,
            0.001449601724743843,
            0.0014162608422338963,
            0.0013836869038641453,
            0.0013518620980903506,
            0.0013207693118602037,
            0.0012903916649520397,
            0.0012607125099748373,
            0.0012317161308601499,
            0.001203386695124209,
            0.0011757088359445333,
            0.0011486675357446074,
            0.001122248126193881,
            0.0010964364046230912,
            0.0010712184011936188,
            0.0010465803788974881,
            0.001022509066388011,
            0.000998991308733821,
        ]

        self.assertTrue(np.allclose(torch_lr_produced, tf_keras_lr_schedule_values))

    def test_static_schedule(self):
        """Test static learning rate schedule returns None."""

        test_weights = [torch.ones([10])]
        lr = 0.01
        train_size = 20000
        epochs = 15

        self.params.train.fp32.number_of_epochs = epochs
        self.params.optimizer.learning_rate_scheduler = "static"
        optimizer = torch.optim.Adam(test_weights, lr=lr)

        lr_sched = get_lr_schedule(
            self.params.train.fp32, optimizer, train_size, self.params
        )

        self.assertIsNone(lr_sched)

    def test_unknown_schedule(self):
        """Test unknown learning rate raises an error."""

        test_weights = [torch.ones([10])]
        lr = 0.01
        train_size = 20000
        epochs = 15

        self.params.train.fp32.number_of_epochs = epochs
        self.params.optimizer.learning_rate_scheduler = "abcd"
        optimizer = torch.optim.Adam(test_weights, lr=lr)

        with self.assertRaises(ValueError):
            get_lr_schedule(self.params.train.fp32, optimizer, train_size, self.params)


if __name__ == "__main__":
    unittest.main()
