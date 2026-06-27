import torch

from pixyz.distributions import Deterministic, Normal
from pixyz.losses import Expectation, Parameter


class TestExpectation:
    def test_sample_mean(self):
        p = Normal(loc=0, scale=1)
        f = p.log_prob()
        e = Expectation(p, f)
        e.eval({}, sample_mean=True)

    def test_vectorized_expectation_preserves_batch_mean(self):
        class Shift(Deterministic):
            def __init__(self):
                super().__init__(var=["z"], cond_var=["x"])

            def forward(self, x):
                return {"z": x + 1}

        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        e = Expectation(Shift(), Parameter("z").mean(), sample_shape=(4,))

        assert torch.equal(e.eval({"x": x}), (x + 1).mean())

    def test_vectorized_expectation_broadcasts_conditional_inputs(self):
        class Encoder(Deterministic):
            def __init__(self):
                super().__init__(var=["z"], cond_var=["x"])

            def forward(self, x):
                return {"z": x + 1}

        class Transition(Deterministic):
            def __init__(self):
                super().__init__(var=["h"], cond_var=["x", "z"])

            def forward(self, x, z):
                return {"h": x + z}

        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        e = Expectation(Encoder() * Transition(), Parameter("h").mean(), sample_shape=(3,))

        assert torch.equal(e.eval({"x": x}), (2 * x + 1).mean())

    def test_vectorized_expectation_broadcasts_non_sampled_conditionals(self):
        class Encoder(Deterministic):
            def __init__(self):
                super().__init__(var=["z"], cond_var=["x"])

            def forward(self, x):
                return {"z": x + 1}

        class Decoder(Normal):
            def __init__(self):
                super().__init__(var=["y"], cond_var=["x", "z"], name="p")

            def forward(self, x, z):
                loc = x + z
                scale = torch.ones_like(loc)
                return {"loc": loc, "scale": scale}

        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        y = torch.zeros_like(x)
        e = Expectation(Encoder(), Decoder().log_prob().mean(), sample_shape=(3,))

        loss = e.eval({"x": x, "y": y})
        assert loss.ndim == 0

    def test_vectorized_expectation_with_auto_gru_cell_batching(self):
        class Encoder(Deterministic):
            def __init__(self):
                super().__init__(var=["z"], cond_var=["x"])

            def forward(self, x):
                return {"z": x + 1}

        class Transition(Deterministic):
            def __init__(self):
                super().__init__(var=["h"], cond_var=["x", "z", "h_prev"])
                self.rnncell = torch.nn.GRUCell(6, 5)

            def forward(self, x, z, h_prev):
                return {"h": self.rnncell(torch.cat((x, z), dim=-1), h_prev)}

        x = torch.randn(2, 3)
        h_prev = torch.zeros(2, 5)
        e = Expectation(Encoder() * Transition(), Parameter("h").mean(), sample_shape=(4,))

        loss = e.eval({"x": x, "h_prev": h_prev})
        assert loss.ndim == 0

    def test_vectorized_expectation_with_auto_lstm_cell_batching(self):
        class Encoder(Deterministic):
            def __init__(self):
                super().__init__(var=["z"], cond_var=["x"])

            def forward(self, x):
                return {"z": x + 1}

        class Transition(Deterministic):
            def __init__(self):
                super().__init__(var=["h", "c"], cond_var=["x", "z", "h_prev", "c_prev"])
                self.lstmcell = torch.nn.LSTMCell(6, 5)

            def forward(self, x, z, h_prev, c_prev):
                h, c = self.lstmcell(torch.cat((x, z), dim=-1), (h_prev, c_prev))
                return {"h": h, "c": c}

        x = torch.randn(2, 3)
        h_prev = torch.zeros(2, 5)
        c_prev = torch.zeros(2, 5)
        e = Expectation(Encoder() * Transition(), Parameter("h").mean(), sample_shape=(4,))

        loss = e.eval({"x": x, "h_prev": h_prev, "c_prev": c_prev})
        assert loss.ndim == 0
