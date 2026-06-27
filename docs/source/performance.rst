Performance Guide
=================

Pixyz is designed around flexible symbolic losses and dictionary-based variable
bindings. That flexibility is useful, but performance-sensitive workloads should
prefer the following patterns.

Recommendations
---------------

- Prefer ``logits`` over ``probs`` for Bernoulli decoders to avoid redundant
  activation and conversion work.
- Use :class:`pixyz.losses.IterativeLoss` with ``series_dim`` set explicitly
  when sequence data is not time-major.
- ``Expectation(..., sample_shape=...)`` is vectorized across the Monte Carlo
  axis, so multiple samples no longer require a Python-side evaluation loop.
- ``GRUCell`` / ``LSTMCell`` / ``RNNCell`` inside
  :class:`pixyz.distributions.Deterministic` are adapted to vectorized sample
  axes automatically. Use :func:`pixyz.utils.call_sample_batch` only for other
  custom modules that still expect flattened batch input.
- Use ``return_all=False`` when intermediate variables do not need to be kept.
- For large models, use ``torch.compile`` through
  :func:`pixyz.utils.compile_if_available` on both distributions and the loss
  object.

Sequence Models
---------------

``IterativeLoss`` is the sequence-loss interface for recurrent objectives and
supports explicit control over the sequence axis.

.. code-block:: python

    from pixyz.losses import IterativeLoss

    loss = IterativeLoss(
        step_loss=step_loss,
        series_var=["x"],
        update_value={"h": "h_prev"},
        timestep_var="t",
        series_dim=0,
    )

For recurrent deterministic transitions, ``GRUCell`` now works with vectorized
Monte Carlo samples without changing the surrounding loss:

.. code-block:: python

    from pixyz.distributions import Deterministic
    class Transition(Deterministic):
        def __init__(self, x_dim, z_dim, h_dim):
            super().__init__(var=["h"], cond_var=["x", "z", "h_prev"])
            self.rnncell = torch.nn.GRUCell(x_dim + z_dim, h_dim)

        def forward(self, x, z, h_prev):
            h = self.rnncell(torch.cat((x, z), dim=-1), h_prev)
            return {"h": h}

Benchmarking
------------

The repository includes a benchmark script for tracking sequence training
performance.

.. code-block:: bash

    python benchmarks/performance.py --device cpu
    python benchmarks/performance.py --device cuda --compile
