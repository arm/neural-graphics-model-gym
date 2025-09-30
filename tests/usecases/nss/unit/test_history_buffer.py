# SPDX-FileCopyrightText: <text>Copyright 2025 Arm Limited and/or
# its affiliates <open-source-office@arm.com></text>
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from ng_model_gym.core.utils.types import HistoryBufferResetFunction
from ng_model_gym.usecases.nss.history_buffer import HistoryBuffer


class TestHistoryBuffer(unittest.TestCase):
    """Test history buffer API"""

    def test_identity_reset_function(self):
        """Test identity reset function"""
        tensor_data = torch.full([2, 4, 128, 128], 1.0, device="cpu")
        history_buffer = HistoryBuffer(
            name="buffer", reset_key="", reset_func=HistoryBufferResetFunction.IDENTITY
        )
        out = history_buffer.reset_func(tensor_data)
        self.assertTrue(torch.equal(out, tensor_data))

    def test_zero_reset_function(self):
        """Test zero reset function"""
        tensor_data = torch.full([2, 4, 128, 128], 3.0, device="cpu")
        history_buffer = HistoryBuffer(
            name="buffer", reset_key="", reset_func=HistoryBufferResetFunction.ZEROS
        )
        out = history_buffer.reset_func(tensor_data)
        self.assertTrue(torch.equal(out, torch.zeros_like(tensor_data)))

    def test_zero_reset_function_with_channel_dim(self):
        """Test zero reset with channel dims specified"""
        tensor_data = torch.full([2, 4, 128, 128], 3.0, device="cpu")
        channel_dims = 2
        history_buffer = HistoryBuffer(
            name="buffer",
            reset_key="",
            reset_func=HistoryBufferResetFunction.ZEROS,
            channel_dim=channel_dims,
        )
        out = history_buffer.reset_func(tensor_data)
        self.assertEqual(out.shape, (2, channel_dims, 128, 128))
        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_ones_reset(self):
        """Test ones reset"""
        tensor_data = torch.full([2, 4, 128, 128], 3.0, device="cpu")
        history_buffer = HistoryBuffer(
            name="buffer", reset_key="", reset_func=HistoryBufferResetFunction.ONES
        )
        out = history_buffer.reset_func(tensor_data)
        self.assertTrue(torch.equal(out, torch.ones_like(out)))

    def test_reset_lr_and_hr(self):
        """Test reset_lr"""
        tensor_data = torch.full([2, 4, 128, 128], 1.0, device="cpu")
        buffer_lr = HistoryBuffer(
            name="lr",
            reset_key="",
            reset_func=HistoryBufferResetFunction.RESET_LR,
            channel_dim=1,
            scale=2,
        )
        out_lr = buffer_lr.reset_func(tensor_data)
        self.assertEqual(out_lr.shape, (2, 1, 64, 64))
        self.assertTrue(torch.equal(out_lr, torch.zeros_like(out_lr)))

    def test_reset_hr(self):
        """test reset_hr"""
        tensor_data = torch.full([2, 4, 128, 128], 1.0, device="cpu")
        buffer_hr = HistoryBuffer(
            name="hr",
            reset_key="",
            reset_func=HistoryBufferResetFunction.RESET_HR,
            channel_dim=1,
            scale=2,
        )
        out_hr = buffer_hr.reset_func(tensor_data)
        self.assertEqual(out_hr.shape, (2, 1, 256, 256))
        self.assertTrue(torch.equal(out_hr, torch.zeros_like(out_hr)))

    def test_custom_reset(self):
        """Test custom reset_func"""
        custom_reset = lambda x: x * 2
        tensor_data = torch.full([2, 4, 128, 128], 1.0, device="cpu")
        buf = HistoryBuffer(name="buffer", reset_key="some", reset_func=custom_reset)
        out = buf.reset_func(tensor_data)
        self.assertTrue(torch.equal(out, tensor_data * 2))

    def test_set_and_get_buffers(self):
        """Test set and get buffers"""
        tensor_data = torch.full([2, 4, 128, 128], 1.0, device="cpu")

        seq = torch.tensor(1)
        channel_dim = 3
        history_buffer = HistoryBuffer(
            name="mock_history",
            reset_key="motion",
            reset_func=HistoryBufferResetFunction.ZEROS,
            update_key="output_linear",
            channel_dim=channel_dim,
        )

        state_tm1 = history_buffer.set(tensor_data, seq)
        self.assertTrue(history_buffer.initialised)
        self.assertEqual(state_tm1.shape, (2, channel_dim, 128, 128))
        self.assertTrue(torch.equal(state_tm1, torch.zeros_like(state_tm1)))

        # Same sequence ID - no resets
        new_tensor_data = torch.full([2, 4, 128, 128], 2.0, device="cpu")
        new_state_tm1 = history_buffer.get(new_tensor_data, seq.clone())
        self.assertTrue(torch.equal(new_state_tm1, state_tm1))
        self.assertTrue(torch.equal(history_buffer.seq_tm1, seq))

        # New sequence
        new_seq = torch.tensor(2)
        self.assertTrue(not torch.equal(history_buffer.seq_tm1, new_seq))
        got = history_buffer.get(new_tensor_data, new_seq)
        self.assertEqual(got.shape, (2, channel_dim, 128, 128))
        self.assertTrue(torch.equal(got, torch.zeros_like(got)))

    def test_update_and_detach(self):
        """Test update and detach buffer"""
        tensor_data = torch.full([2, 4, 128, 128], 5.0, device="cpu")
        seq = torch.tensor(1)
        history_buffer = HistoryBuffer(
            name="buffer", reset_key="", reset_func=HistoryBufferResetFunction.IDENTITY
        )
        history_buffer.set(tensor_data, seq)

        # Update
        new_state = torch.full([2, 4, 128, 128], 10.0, device="cpu")
        history_buffer.update(new_state)
        self.assertTrue(torch.equal(history_buffer.state_tm1, new_state))

        # Detach
        old = history_buffer.state_tm1
        history_buffer.detach()
        self.assertIsNot(history_buffer.state_tm1, old)


if __name__ == "__main__":
    unittest.main()
