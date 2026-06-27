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
