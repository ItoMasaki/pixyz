from copy import deepcopy

import sympy

from .losses import Loss
from ..utils import get_dict_values


class IterativeLoss(Loss):
    r"""
    Iterative loss.

    This class allows implementing an arbitrary model which requires iteration.

    .. math::

        \mathcal{L} = \sum_{t=0}^{T-1}\mathcal{L}_{step}(x_t, h_t),

    where :math:`x_t = f_{slice\_step}(x, t)`.

    Examples
    --------
    >>> import torch
    >>> from torch.nn import functional as F
    >>> from pixyz.distributions import Normal, Bernoulli, Deterministic
    >>>
    >>> # Set distributions
    >>> x_dim = 128
    >>> z_dim = 64
    >>> h_dim = 32
    >>>
    >>> # p(x|z,h_{prev})
    >>> class Decoder(Bernoulli):
    ...     def __init__(self):
    ...         super().__init__(var=["x"],cond_var=["z", "h_prev"],name="p")
    ...         self.fc = torch.nn.Linear(z_dim + h_dim, x_dim)
    ...     def forward(self, z, h_prev):
    ...         return {"probs": torch.sigmoid(self.fc(torch.cat((z, h_prev), dim=-1)))}
    ...
    >>> # q(z|x,h_{prev})
    >>> class Encoder(Normal):
    ...     def __init__(self):
    ...         super().__init__(var=["z"],cond_var=["x", "h_prev"],name="q")
    ...         self.fc_loc = torch.nn.Linear(x_dim + h_dim, z_dim)
    ...         self.fc_scale = torch.nn.Linear(x_dim + h_dim, z_dim)
    ...     def forward(self, x, h_prev):
    ...         xh = torch.cat((x, h_prev), dim=-1)
    ...         return {"loc": self.fc_loc(xh), "scale": F.softplus(self.fc_scale(xh))}
    ...
    >>> # f(h|x,z,h_{prev}) (update h)
    >>> class Recurrence(Deterministic):
    ...     def __init__(self):
    ...         super().__init__(var=["h"], cond_var=["x", "z", "h_prev"], name="f")
    ...         self.rnncell = torch.nn.GRUCell(x_dim + z_dim, h_dim)
    ...     def forward(self, x, z, h_prev):
    ...         return {"h": self.rnncell(torch.cat((z, x), dim=-1), h_prev)}
    >>>
    >>> p = Decoder()
    >>> q = Encoder()
    >>> f = Recurrence()
    >>>
    >>> # Set the loss class
    >>> step_loss_cls = p.log_prob().expectation(q * f).mean()
    >>> print(step_loss_cls)
    mean \left(\mathbb{E}_{q(z,h|x,h_{prev})} \left[\log p(x|z,h_{prev}) \right] \right)
    >>> loss_cls = IterativeLoss(step_loss=step_loss_cls,
    ...                          series_var=["x"], update_value={"h": "h_prev"})
    >>> print(loss_cls)
    \sum_{t=0}^{t_{max} - 1} mean \left(\mathbb{E}_{q(z,h|x,h_{prev})} \left[\log p(x|z,h_{prev}) \right] \right)
    >>>
    >>> # Evaluate
    >>> x_sample = torch.randn(30, 2, 128) # (timestep_size, batch_size, feature_size)
    >>> h_init = torch.zeros(2, 32) # (batch_size, h_dim)
    >>> loss = loss_cls.eval({"x": x_sample, "h_prev": h_init})
    >>> print(loss) # doctest: +SKIP
    tensor(-2826.0906, grad_fn=<AddBackward0>
    """

    def __init__(self, step_loss, max_iter=None,
                 series_var=(), update_value={}, slice_step=None, timestep_var=(), series_dim=0):
        super().__init__()
        self.step_loss = step_loss
        self.max_iter = max_iter
        self.update_value = update_value
        self.series_var = tuple(series_var)
        self.series_dim = series_dim
        self._update_pairs = tuple(update_value.items())
        self._update_sources = tuple(update_value.keys())
        self._update_targets = tuple(update_value.values())

        if isinstance(timestep_var, str):
            self.timestep_var = timestep_var
        elif timestep_var:
            self.timestep_var = timestep_var[0]
        else:
            self.timestep_var = None

        if self.timestep_var:
            self.timpstep_symbol = sympy.Symbol(self.timestep_var)
        else:
            self.timpstep_symbol = sympy.Symbol("t")

        if not self.series_var and (max_iter is None):
            raise ValueError()

        self.slice_step = slice_step
        if self.slice_step:
            self.step_loss = self.step_loss.expectation(self.slice_step)

        _input_var = []
        _input_var += deepcopy(self.step_loss.input_var)
        _input_var += list(self.series_var)
        _input_var += list(self._update_targets)

        self._input_var = sorted(set(_input_var), key=_input_var.index)

        if self.timestep_var and self.timestep_var in self._input_var:
            self._input_var.remove(self.timestep_var)

    @property
    def _symbol(self):
        dummy_loss = sympy.Symbol("dummy_loss")
        if self.max_iter:
            max_iter = self.max_iter
        else:
            max_iter = sympy.Symbol(sympy.latex(self.timpstep_symbol) + "_{max}")

        _symbol = sympy.Sum(dummy_loss, (self.timpstep_symbol, 0, max_iter - 1))
        _symbol = _symbol.subs({dummy_loss: self.step_loss._symbol})
        return _symbol

    def slice_step_fn(self, t, x):
        return {k: v.select(self.series_dim, t) for k, v in x.items()}

    def _update_loop_state(self, x_dict):
        if not self._update_pairs:
            return

        replaced_values = {}
        for source_key, target_key in self._update_pairs:
            if source_key in x_dict:
                replaced_values[target_key] = x_dict[source_key]

        for source_key, target_key in self._update_pairs:
            if source_key != target_key and source_key in x_dict:
                del x_dict[source_key]

        x_dict.update(replaced_values)

    def forward(self, x_dict, **kwargs):
        series_x_dict = get_dict_values(x_dict, self.series_var, return_dict=True)

        if self.max_iter:
            max_iter = self.max_iter
        else:
            max_iter = series_x_dict[self.series_var[0]].shape[self.series_dim]

        mask = kwargs.get("mask")
        if mask is not None:
            mask = mask.float()

        timestep_var = self.timestep_var
        step_loss_fn = self.step_loss
        step_loss_sum = 0

        if self.slice_step:
            for t in range(max_iter):
                if timestep_var:
                    x_dict[timestep_var] = t

                step_loss, samples = step_loss_fn(x_dict, **kwargs)
                x_dict.update(samples)
                if mask is not None:
                    step_loss = step_loss * mask[t]
                step_loss_sum += step_loss
                self._update_loop_state(x_dict)
        else:
            series_vars = self.series_var
            for t in range(max_iter):
                if timestep_var:
                    x_dict[timestep_var] = t

                for var_name in series_vars:
                    x_dict[var_name] = series_x_dict[var_name].select(self.series_dim, t)

                step_loss, samples = step_loss_fn(x_dict, **kwargs)
                x_dict.update(samples)
                if mask is not None:
                    step_loss = step_loss * mask[t]
                step_loss_sum += step_loss
                self._update_loop_state(x_dict)

        x_dict.update(series_x_dict)
        return step_loss_sum, x_dict


class SequentialLoss(IterativeLoss):
    r"""
    High-level wrapper for recurrent or sequential objectives.

    This class is equivalent to :class:`IterativeLoss` but uses sequence-oriented
    argument names and supports choosing the sequence axis explicitly.

    Parameters
    ----------
    step_loss : pixyz.losses.Loss
        Loss evaluated at each time step.
    sequence_var : list or tuple
        Variables that contain a time dimension.
    state_update : dict
        Mapping from newly generated state variables to the state variables used
        at the next step. For example, ``{"h": "h_prev"}``.
    max_steps : int, defaults to None
        Fixed number of steps. If omitted, this is inferred from the first
        sequence variable.
    slice_step : pixyz.distributions.Distribution, defaults to None
        Optional slicing distribution used by the legacy iterative interface.
    time_var : str or tuple, defaults to ()
        Optional time-step variable injected into the step loss.
    sequence_dim : int, defaults to 0
        Axis that corresponds to the time dimension in ``sequence_var``.
    """

    def __init__(self, step_loss, sequence_var=(), state_update=None, max_steps=None,
                 slice_step=None, time_var=(), sequence_dim=0):
        if state_update is None:
            state_update = {}
        super().__init__(step_loss=step_loss, max_iter=max_steps,
                         series_var=sequence_var, update_value=state_update,
                         slice_step=slice_step, timestep_var=time_var, series_dim=sequence_dim)
