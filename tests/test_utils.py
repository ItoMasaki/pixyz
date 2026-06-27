import torch

from pixyz.utils import call_sample_batch


class TestSampleBatchHelpers:
    def test_call_sample_batch_gru_cell_matches_manual_flatten(self):
        cell = torch.nn.GRUCell(7, 5)
        x = torch.randn(3, 4, 7)
        h = torch.randn(3, 4, 5)

        actual = call_sample_batch(cell, x, h)
        expected = cell(x.reshape(-1, 7), h.reshape(-1, 5)).reshape(3, 4, 5)

        assert torch.allclose(actual, expected)

    def test_call_sample_batch_lstm_cell_matches_manual_flatten(self):
        cell = torch.nn.LSTMCell(7, 5)
        x = torch.randn(2, 3, 4, 7)
        h = torch.randn(2, 3, 4, 5)
        c = torch.randn(2, 3, 4, 5)

        actual_h, actual_c = call_sample_batch(cell, x, (h, c))
        expected_h, expected_c = cell(x.reshape(-1, 7), (h.reshape(-1, 5), c.reshape(-1, 5)))
        expected_h = expected_h.reshape(2, 3, 4, 5)
        expected_c = expected_c.reshape(2, 3, 4, 5)

        assert torch.allclose(actual_h, expected_h)
        assert torch.allclose(actual_c, expected_c)
