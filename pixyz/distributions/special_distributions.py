from __future__ import print_function

import torch

from .distributions import Distribution
from ..utils import broadcast_sample_dict, call_sample_batch


_AUTO_SAMPLE_BATCH_MODULE_TYPES = (
    torch.nn.RNNCell,
    torch.nn.GRUCell,
    torch.nn.LSTMCell,
)


class _SampleBatchModuleProxy:
    def __init__(self, module, trailing_ndims):
        self._module = module
        self._trailing_ndims = trailing_ndims

    def __call__(self, *args, **kwargs):
        return call_sample_batch(self._module, *args, trailing_ndims=self._trailing_ndims, **kwargs)

    def __getattr__(self, item):
        return getattr(self._module, item)

    def __repr__(self):
        return repr(self._module)


class Deterministic(Distribution):
    """
    Deterministic distribution (or degeneration distribution)

    Examples
    --------
    >>> import torch
    >>> class Generator(Deterministic):
    ...     def __init__(self):
    ...         super().__init__(var=["x"], cond_var=["z"])
    ...         self.model = torch.nn.Linear(64, 512)
    ...     def forward(self, z):
    ...         return {"x": self.model(z)}
    >>> p = Generator()
    >>> print(p)
    Distribution:
      p(x|z)
    Network architecture:
      Generator(
        name=p, distribution_name=Deterministic,
        var=['x'], cond_var=['z'], input_var=['z'], features_shape=torch.Size([])
        (model): Linear(in_features=64, out_features=512, bias=True)
      )
    >>> sample = p.sample({"z": torch.randn(1, 64)})
    >>> p.log_prob().eval(sample) # log_prob is not defined.
    Traceback (most recent call last):
     ...
    NotImplementedError: Log probability of deterministic distribution is not defined.
    """

    def __init__(self, var, cond_var=[], name='p', **kwargs):
        super().__init__(var=var, cond_var=cond_var, name=name, **kwargs)
        self._sample_batch_module_specs = {}
        self._sample_batch_module_proxies = {}

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, _AUTO_SAMPLE_BATCH_MODULE_TYPES):
            specs = self.__dict__.get("_sample_batch_module_specs")
            if specs is not None:
                specs.setdefault(name, 1)
            proxies = self.__dict__.get("_sample_batch_module_proxies")
            if proxies is not None:
                proxies.pop(name, None)

    def __getattr__(self, item):
        attr = super().__getattr__(item)
        specs = self.__dict__.get("_sample_batch_module_specs")
        if specs and item in specs:
            proxies = self.__dict__.get("_sample_batch_module_proxies")
            if item not in proxies:
                proxies[item] = _SampleBatchModuleProxy(attr, specs[item])
            return proxies[item]
        return attr

    def register_sample_batch_module(self, name, trailing_ndims=1):
        self._sample_batch_module_specs[name] = trailing_ndims
        self._sample_batch_module_proxies.pop(name, None)

    def unregister_sample_batch_module(self, name):
        self._sample_batch_module_specs.pop(name, None)
        self._sample_batch_module_proxies.pop(name, None)

    @property
    def distribution_name(self):
        return "Deterministic"

    def sample(self, x_dict={}, return_all=True, **kwargs):
        input_dict = self._get_input_dict(x_dict)
        sample_shape = torch.Size(kwargs.get("sample_shape", torch.Size()))

        if sample_shape:
            input_dict = broadcast_sample_dict(input_dict, sample_shape)

        output_dict = self.forward(**input_dict)

        if set(output_dict.keys()) != set(self._var):
            raise ValueError("Output variables are not the same as `var`.")

        if return_all:
            if not x_dict:
                return output_dict
            x_dict = x_dict.copy()
            x_dict.update(output_dict)
            return x_dict

        return output_dict

    def sample_mean(self, x_dict):
        return self.sample(x_dict, return_all=False)[self._var[0]]

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None, **kwargs):
        raise NotImplementedError("Log probability of deterministic distribution is not defined.")

    @property
    def has_reparam(self):
        return True


class EmpiricalDistribution(Distribution):
    """
    Data distribution.

    Samples from this distribution equal given inputs.

    Examples
    --------
    >>> import torch
    >>> p = EmpiricalDistribution(var=["x"])
    >>> print(p)
    Distribution:
      p_{data}(x)
    Network architecture:
      EmpiricalDistribution(
        name=p_{data}, distribution_name=Data distribution,
        var=['x'], cond_var=[], input_var=['x'], features_shape=torch.Size([])
      )
    >>> sample = p.sample({"x": torch.randn(1, 64)})
    """

    def __init__(self, var, name="p_{data}"):
        super().__init__(var=var, cond_var=[], name=name)

    @property
    def distribution_name(self):
        return "Data distribution"

    def sample(self, x_dict={}, return_all=True, **kwargs):
        output_dict = self._get_input_dict(x_dict)

        if return_all:
            if not x_dict:
                return output_dict
            x_dict = x_dict.copy()
            x_dict.update(output_dict)
            return x_dict
        return output_dict

    def sample_mean(self, x_dict):
        return self.sample(x_dict, return_all=False)[self._var[0]]

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None, **kwargs):
        raise NotImplementedError()

    @property
    def input_var(self):
        """
        In EmpiricalDistribution, `input_var` is same as `var`.
        """

        return self.var

    @property
    def has_reparam(self):
        return True
