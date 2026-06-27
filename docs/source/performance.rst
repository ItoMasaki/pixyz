Performance Guide
=================

Pixyz is designed around flexible symbolic losses and dictionary-based variable
bindings. That flexibility is useful, but performance-sensitive workloads should
prefer the following patterns.

Recommendations
---------------

- Prefer ``logits`` over ``probs`` for Bernoulli decoders to avoid redundant
  activation and conversion work.
- Use :class:`pixyz.losses.SequentialLoss` or :class:`pixyz.losses.IterativeLoss`
  with ``series_dim`` set explicitly when sequence data is not time-major.
- ``Expectation(..., sample_shape=...)`` is vectorized across the Monte Carlo
  axis, so multiple samples no longer require a Python-side evaluation loop.
- Use ``return_all=False`` when intermediate variables do not need to be kept.
- For large models, use ``torch.compile`` through
  :func:`pixyz.utils.compile_if_available` on both distributions and the loss
  object.

Sequence Models
---------------

``SequentialLoss`` is the preferred high-level interface for recurrent losses.
It gives the same execution semantics as ``IterativeLoss`` with clearer names
 and explicit control over the sequence axis.

.. code-block:: python

    from pixyz.losses import SequentialLoss

    loss = SequentialLoss(
        step_loss=step_loss,
        sequence_var=["x"],
        state_update={"h": "h_prev"},
        time_var="t",
        sequence_dim=0,
    )

Benchmarking
------------

The repository includes a benchmark script for tracking sequence training
performance.

.. code-block:: bash

    python benchmarks/performance.py --device cpu
    python benchmarks/performance.py --device cuda --compile
