import math
import torch
from torch.distributions import Normal as NormalTorch
from torch.distributions import Bernoulli as BernoulliTorch
from torch.distributions import RelaxedBernoulli as RelaxedBernoulliTorch
from torch.distributions import RelaxedOneHotCategorical as RelaxedOneHotCategoricalTorch
from torch.distributions.one_hot_categorical import OneHotCategorical as CategoricalTorch
from torch.distributions import Multinomial as MultinomialTorch
from torch.distributions import Dirichlet as DirichletTorch
from torch.distributions import Beta as BetaTorch
from torch.distributions import Laplace as LaplaceTorch
from torch.distributions import Gamma as GammaTorch
from torch.distributions.utils import broadcast_all, probs_to_logits
from torch.nn.functional import binary_cross_entropy_with_logits

from ..utils import get_dict_values, sum_samples
from .distributions import DistributionBase


def _valid_param_dict(raw_dict):
    return {var_name: value for var_name, value in raw_dict.items() if value is not None}


class Normal(DistributionBase):
    """Normal distribution parameterized by :attr:`loc` and :attr:`scale`. """
    def __init__(self, var=['x'], cond_var=[], name='p', features_shape=torch.Size(), loc=None, scale=None):
        super().__init__(var, cond_var, name, features_shape, **_valid_param_dict({'loc': loc, 'scale': scale}))

    @property
    def params_keys(self):
        return ["loc", "scale"]

    @property
    def distribution_torch_class(self):
        return NormalTorch

    @property
    def distribution_name(self):
        return "Normal"

    @property
    def has_reparam(self):
        return True

    @staticmethod
    def _coerce_param_pair(loc, scale):
        if torch.is_tensor(loc):
            ref = loc
        elif torch.is_tensor(scale):
            ref = scale
        else:
            ref = None

        if not torch.is_tensor(loc):
            if ref is None:
                loc = torch.tensor(loc, dtype=torch.float)
            else:
                loc = torch.as_tensor(loc, dtype=ref.dtype, device=ref.device)
        if not torch.is_tensor(scale):
            scale = torch.as_tensor(scale, dtype=loc.dtype, device=loc.device)
        return loc, scale

    @staticmethod
    def _expand_batch_params(batch_n, *params):
        if not batch_n:
            return params

        batch_shape = params[0].shape
        if batch_shape[0] == 1:
            expand_shape = torch.Size([batch_n]) + batch_shape[1:]
            return tuple(param.expand(expand_shape) for param in params)
        if batch_shape[0] == batch_n:
            return params
        raise ValueError(f"Batch shape mismatch. batch_shape from parameters: {batch_shape}\n"
                         f" specified batch size:{batch_n}")

    @staticmethod
    def _expand_sample_param(param, sample_shape):
        if sample_shape == torch.Size():
            return param
        return param.expand(sample_shape + param.shape)

    def _sample_from_params(self, params, batch_n=None, sample_shape=torch.Size(), reparam=False, sample_mean=False):
        loc, scale = self._coerce_param_pair(params["loc"], params["scale"])
        loc, scale = self._expand_batch_params(batch_n, loc, scale)

        if sample_mean:
            return self._expand_sample_param(loc, torch.Size(sample_shape))
        else:
            sample_loc = self._expand_sample_param(loc, torch.Size(sample_shape))
            sample_scale = self._expand_sample_param(scale, torch.Size(sample_shape))
            noise = torch.randn_like(sample_scale)
            return sample_loc + sample_scale * noise

    def _log_prob_from_params(self, params, x_targets, sum_features=True, feature_dims=None):
        [x_target] = x_targets
        loc, scale = self._coerce_param_pair(params["loc"], params["scale"])
        loc, scale, x_target = torch.broadcast_tensors(loc, scale, x_target)
        variance = scale.pow(2)
        log_prob = -0.5 * ((x_target - loc) ** 2) / variance - scale.log() - 0.5 * math.log(2. * math.pi)
        if sum_features:
            log_prob = sum_samples(log_prob, feature_dims)
        return log_prob

    def _entropy_from_params(self, params, sum_features=True, feature_dims=None):
        _, scale = self._coerce_param_pair(0., params["scale"])
        entropy = scale.log() + 0.5 * (1.0 + math.log(2. * math.pi))
        if sum_features:
            entropy = sum_samples(entropy, feature_dims)
        return entropy

    def _sample_mean_from_params(self, params):
        loc, _ = self._coerce_param_pair(params["loc"], params["scale"])
        return loc

    def _sample_variance_from_params(self, params):
        _, scale = self._coerce_param_pair(params["loc"], params["scale"])
        return scale.pow(2)


class BernoulliTorchOld(BernoulliTorch):
    def log_prob(self, value):
        logits, value = broadcast_all(self.logits, value)
        return -binary_cross_entropy_with_logits(logits, value, reduction='none')


class Bernoulli(DistributionBase):
    """Bernoulli distribution parameterized by :attr:`probs`."""
    def __init__(self, var=['x'], cond_var=[], name='p', features_shape=torch.Size(), probs=None, logits=None):
        if probs is not None and logits is not None:
            raise ValueError("Specify either probs or logits, not both.")
        super().__init__(var, cond_var, name, features_shape, **_valid_param_dict({'probs': probs, 'logits': logits}))

    @property
    def params_keys(self):
        return ["logits"] if "logits" in self._buffers else ["probs"]

    @property
    def distribution_torch_class(self):
        return BernoulliTorchOld

    @property
    def distribution_name(self):
        return "Bernoulli"

    @property
    def has_reparam(self):
        return False

    def _validate_params(self, params):
        if "logits" in params or "probs" in params:
            return params
        raise ValueError(f"{type(self)} class requires following parameters: probs or logits\n"
                         f"but got {set(params.keys())}")

    @staticmethod
    def _expand_batch_param(batch_n, param):
        if not batch_n:
            return param

        batch_shape = param.shape
        if batch_shape[0] == 1:
            return param.expand(torch.Size([batch_n]) + batch_shape[1:])
        if batch_shape[0] == batch_n:
            return param
        raise ValueError(f"Batch shape mismatch. batch_shape from parameters: {batch_shape}\n"
                         f" specified batch size:{batch_n}")

    @staticmethod
    def _expand_sample_param(param, sample_shape):
        if sample_shape == torch.Size():
            return param
        return param.expand(sample_shape + param.shape)

    def _get_logits_and_probs(self, params):
        logits = params.get("logits")
        probs = params.get("probs")
        if logits is not None and not torch.is_tensor(logits):
            logits = torch.tensor(logits, dtype=torch.float)
        if probs is not None and not torch.is_tensor(probs):
            if logits is None:
                probs = torch.tensor(probs, dtype=torch.float)
            else:
                probs = torch.as_tensor(probs, dtype=logits.dtype, device=logits.device)
        if logits is None:
            logits = probs_to_logits(probs, is_binary=True)
        elif probs is None:
            probs = torch.sigmoid(logits)
        return logits, probs

    def _get_probs(self, params):
        probs = params.get("probs")
        logits = params.get("logits")
        if probs is None:
            if not torch.is_tensor(logits):
                logits = torch.tensor(logits, dtype=torch.float)
            return torch.sigmoid(logits)
        if not torch.is_tensor(probs):
            return torch.tensor(probs, dtype=torch.float)
        return probs

    def set_dist(self, x_dict={}, batch_n=None, **kwargs):
        params = self._validate_params(self._resolve_params(x_dict, **kwargs))
        logits, probs = self._get_logits_and_probs(params)
        if "logits" in params:
            self._dist = self.distribution_torch_class(logits=logits)
        else:
            self._dist = self.distribution_torch_class(probs=probs)

        self._expand_dist_batch(batch_n)

    def _sample_from_params(self, params, batch_n=None, sample_shape=torch.Size(), reparam=False, sample_mean=False):
        probs = self._get_probs(params)
        probs = self._expand_batch_param(batch_n, probs)

        if sample_mean:
            return self._expand_sample_param(probs, torch.Size(sample_shape))
        else:
            probs = self._expand_sample_param(probs, torch.Size(sample_shape))
            return torch.bernoulli(probs)

    def _log_prob_from_params(self, params, x_targets, sum_features=True, feature_dims=None):
        [x_target] = x_targets
        if "logits" in params:
            logits = params["logits"]
            if not torch.is_tensor(logits):
                logits = torch.tensor(logits, dtype=torch.float)
            logits, x_target = torch.broadcast_tensors(logits, x_target)
            log_prob = -binary_cross_entropy_with_logits(logits, x_target, reduction='none')
        else:
            probs = params["probs"]
            if not torch.is_tensor(probs):
                probs = torch.tensor(probs, dtype=torch.float)
            probs, x_target = torch.broadcast_tensors(probs, x_target)
            probs = probs.clamp(torch.finfo(probs.dtype).tiny, 1. - torch.finfo(probs.dtype).eps)
            log_prob = x_target * torch.log(probs) + (1. - x_target) * torch.log1p(-probs)
        if sum_features:
            log_prob = sum_samples(log_prob, feature_dims)
        return log_prob

    def _entropy_from_params(self, params, sum_features=True, feature_dims=None):
        _, probs = self._get_logits_and_probs(params)
        probs = probs.clamp(1e-6, 1. - 1e-6)
        entropy = -(probs * probs.log() + (1. - probs) * torch.log1p(-probs))
        if sum_features:
            entropy = sum_samples(entropy, feature_dims)
        return entropy

    def _sample_mean_from_params(self, params):
        return self._get_probs(params)

    def _sample_variance_from_params(self, params):
        probs = self._get_probs(params)
        return probs * (1. - probs)


class RelaxedBernoulli(Bernoulli):
    """Relaxed (re-parameterizable) Bernoulli distribution parameterized by :attr:`probs` and :attr:`temperature`."""
    def __init__(self, var=["x"], cond_var=[], name="p", features_shape=torch.Size(), temperature=torch.tensor(0.1),
                 probs=None):
        super(Bernoulli, self).__init__(var, cond_var, name, features_shape, **_valid_param_dict({
            'probs': probs, 'temperature': temperature}))

    @property
    def params_keys(self):
        return ["probs", "temperature"]

    @property
    def distribution_torch_class(self):
        """Use relaxed version only when sampling"""
        return RelaxedBernoulliTorch

    @property
    def distribution_name(self):
        return "RelaxedBernoulli"

    def set_dist(self, x_dict={}, batch_n=None, sampling=False, **kwargs):
        """Set :attr:`dist` as PyTorch distributions given parameters.

        This requires that :attr:`params_keys` and :attr:`distribution_torch_class` are set.

        Parameters
        ----------
        x_dict : :obj:`dict`, defaults to {}.
            Parameters of this distribution.
        batch_n : :obj:`int`, defaults to None.
            Set batch size of parameters.
        sampling : :obj:`bool` defaults to False.
            If it is false, the distribution will not be relaxed to compute log_prob.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        params = self.get_params(x_dict, **kwargs)
        if set(self.params_keys) != set(params.keys()):
            raise ValueError("{} class requires following parameters: {}\n"
                             "but got {}".format(type(self), set(self.params_keys), set(params.keys())))

        if sampling:
            self._dist = self.distribution_torch_class(**params)
        else:
            hard_params_keys = ["probs"]
            self._dist = BernoulliTorchOld(**get_dict_values(params, hard_params_keys, return_dict=True))

        # expand batch_n
        if batch_n:
            batch_shape = self._dist.batch_shape
            if batch_shape[0] == 1:
                self._dist = self._dist.expand(torch.Size([batch_n]) + batch_shape[1:])
            elif batch_shape[0] == batch_n:
                return
            else:
                raise ValueError()

    def sample(self, x_dict={}, batch_n=None, sample_shape=torch.Size(), return_all=True, reparam=False,
               sample_mean=False, **kwargs):
        # check whether the input is valid or convert it to valid dictionary.
        input_dict = self._get_input_dict(x_dict)

        self.set_dist(input_dict, batch_n=batch_n, sampling=True)
        if sample_mean:
            mean = self.dist.mean
            if sample_shape != torch.Size():
                unsqueeze_shape = torch.Size([1] * len(sample_shape))
                unrepeat_shape = torch.Size([1] * mean.ndim)
                mean = mean.reshape(unsqueeze_shape + mean.shape).repeat(sample_shape + unrepeat_shape)
            output_dict = {self._var[0]: mean}
        else:
            output_dict = self.get_sample(reparam=reparam, sample_shape=sample_shape)

        if return_all:
            x_dict = x_dict.copy()
            x_dict.update(output_dict)
            return x_dict

        return output_dict

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None, **kwargs):
        _x_dict = get_dict_values(x_dict, self._cond_var, return_dict=True)
        self.set_dist(_x_dict, sampling=False, **kwargs)

        x_targets = get_dict_values(x_dict, self._var)
        if len(x_targets) == 0:
            raise ValueError(f"x_dict has no value of the stochastic variable. x_dict: {x_dict}")
        log_prob = self.dist.log_prob(*x_targets)
        if sum_features:
            log_prob = sum_samples(log_prob, feature_dims)

        return log_prob

    @property
    def has_reparam(self):
        return True


class FactorizedBernoulli(Bernoulli):
    """
    Factorized Bernoulli distribution parameterized by :attr:`probs`.

    References
    ----------
    [Vedantam+ 2017] Generative Models of Visually Grounded Imagination

    """
    def __init__(self, var=['x'], cond_var=[], name='p', features_shape=torch.Size(), probs=None):
        super().__init__(var=var, cond_var=cond_var, name=name, features_shape=features_shape, probs=probs)

    @property
    def distribution_name(self):
        return "FactorizedBernoulli"

    def get_log_prob(self, x_dict, sum_features=True, feature_dims=None, **kwargs):
        log_prob = super().get_log_prob(x_dict, sum_features=False, **kwargs)
        [_x] = get_dict_values(x_dict, self._var)
        log_prob[_x == 0] = 0
        if sum_features:
            log_prob = sum_samples(log_prob, feature_dims)
        return log_prob


class CategoricalTorchOld(CategoricalTorch):
    def log_prob(self, value):
        indices = value.max(-1)[1]
        return self._categorical.log_prob(indices)


class Categorical(DistributionBase):
    """Categorical distribution parameterized by :attr:`probs`."""
    def __init__(self, var=['x'], cond_var=[], name='p', features_shape=torch.Size(), probs=None):
        super().__init__(var=var, cond_var=cond_var, name=name, features_shape=features_shape,
                         **_valid_param_dict({'probs': probs}))

    @property
    def params_keys(self):
        return ["probs"]

    @property
    def distribution_torch_class(self):
        return CategoricalTorchOld

    @property
    def distribution_name(self):
        return "Categorical"

    @property
    def has_reparam(self):
        return False


class RelaxedCategorical(Categorical):
    """
    Relaxed (re-parameterizable) categorical distribution parameterized by :attr:`probs` and :attr:`temperature`.
    Notes: a shape of temperature should contain the event shape of this Categorical distribution.
    """
    def __init__(self, var=["x"], cond_var=[], name="p", features_shape=torch.Size(), temperature=torch.tensor(0.1),
                 probs=None):
        super(Categorical, self).__init__(var, cond_var, name, features_shape,
                                          **_valid_param_dict({'probs': probs, 'temperature': temperature}))

    @property
    def params_keys(self):
        return ['probs', 'temperature']

    @property
    def distribution_torch_class(self):
        """Use relaxed version only when sampling"""
        return RelaxedOneHotCategoricalTorch

    @property
    def distribution_name(self):
        return "RelaxedCategorical"

    def set_dist(self, x_dict={}, batch_n=None, sampling=False, **kwargs):
        """Set :attr:`dist` as PyTorch distributions given parameters.

        This requires that :attr:`params_keys` and :attr:`distribution_torch_class` are set.

        Parameters
        ----------
        x_dict : :obj:`dict`, defaults to {}.
            Parameters of this distribution.
        batch_n : :obj:`int`, defaults to None.
            Set batch size of parameters.
        sampling : :obj:`bool` defaults to False.
            If it is false, the distribution will not be relaxed to compute log_prob.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------

        """
        params = self.get_params(x_dict, **kwargs)
        if set(self.params_keys) != set(params.keys()):
            raise ValueError("{} class requires following parameters: {}\n"
                             "but got {}".format(type(self), set(self.params_keys), set(params.keys())))

        if sampling:
            self._dist = self.distribution_torch_class(**params)
        else:
            hard_params_keys = ["probs"]
            self._dist = BernoulliTorchOld(**get_dict_values(params, hard_params_keys, return_dict=True))

        # expand batch_n
        if batch_n:
            batch_shape = self._dist.batch_shape
            if batch_shape[0] == 1:
                self._dist = self._dist.expand(torch.Size([batch_n]) + batch_shape[1:])
            elif batch_shape[0] == batch_n:
                return
            else:
                raise ValueError()

    def sample(self, x_dict={}, batch_n=None, sample_shape=torch.Size(), return_all=True, reparam=False,
               sample_mean=False, **kwargs):
        # check whether the input is valid or convert it to valid dictionary.
        input_dict = self._get_input_dict(x_dict)

        self.set_dist(input_dict, batch_n=batch_n, sampling=True)
        if sample_mean:
            mean = self.dist.mean
            if sample_shape != torch.Size():
                unsqueeze_shape = torch.Size([1] * len(sample_shape))
                unrepeat_shape = torch.Size([1] * mean.ndim)
                mean = mean.reshape(unsqueeze_shape + mean.shape).repeat(sample_shape + unrepeat_shape)
            output_dict = {self._var[0]: mean}
        else:
            output_dict = self.get_sample(reparam=reparam, sample_shape=sample_shape)

        if return_all:
            x_dict = x_dict.copy()
            x_dict.update(output_dict)
            return x_dict

        return output_dict

    @property
    def has_reparam(self):
        return True


class Multinomial(DistributionBase):
    """Multinomial distribution parameterized by :attr:`total_count` and :attr:`probs`."""

    def __init__(self, total_count=1, var=["x"], cond_var=[], name="p", features_shape=torch.Size(), probs=None):
        self._total_count = total_count

        super().__init__(var=var, cond_var=cond_var, name=name, features_shape=features_shape,
                         **_valid_param_dict({'probs': probs}))

    @property
    def total_count(self):
        return self._total_count

    @property
    def params_keys(self):
        return ["probs"]

    @property
    def distribution_torch_class(self):
        return MultinomialTorch

    @property
    def distribution_name(self):
        return "Multinomial"

    @property
    def has_reparam(self):
        return False


class Dirichlet(DistributionBase):
    """Dirichlet distribution parameterized by :attr:`concentration`."""
    def __init__(self, var=["x"], cond_var=[], name="p", features_shape=torch.Size(), concentration=None):
        super().__init__(var=var, cond_var=cond_var, name=name, features_shape=features_shape,
                         **_valid_param_dict({'concentration': concentration}))

    @property
    def params_keys(self):
        return ["concentration"]

    @property
    def distribution_torch_class(self):
        return DirichletTorch

    @property
    def distribution_name(self):
        return "Dirichlet"

    @property
    def has_reparam(self):
        return True


class Beta(DistributionBase):
    """Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0`."""
    def __init__(self, var=["x"], cond_var=[], name="p", features_shape=torch.Size(), concentration1=None,
                 concentration0=None):
        super().__init__(var=var, cond_var=cond_var, name=name, features_shape=features_shape,
                         **_valid_param_dict({'concentration1': concentration1, 'concentration0': concentration0}))

    @property
    def params_keys(self):
        return ["concentration1", "concentration0"]

    @property
    def distribution_torch_class(self):
        return BetaTorch

    @property
    def distribution_name(self):
        return "Beta"

    @property
    def has_reparam(self):
        return True


class Laplace(DistributionBase):
    """
    Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.
    """
    def __init__(self, var=["x"], cond_var=[], name="p", features_shape=torch.Size(), loc=None, scale=None):
        super().__init__(var=var, cond_var=cond_var, name=name, features_shape=features_shape,
                         **_valid_param_dict({'loc': loc, 'scale': scale}))

    @property
    def params_keys(self):
        return ["loc", "scale"]

    @property
    def distribution_torch_class(self):
        return LaplaceTorch

    @property
    def distribution_name(self):
        return "Laplace"

    @property
    def has_reparam(self):
        return True


class Gamma(DistributionBase):
    """
    Gamma distribution parameterized by :attr:`concentration` and :attr:`rate`.
    """
    def __init__(self, var=["x"], cond_var=[], name="p", features_shape=torch.Size(), concentration=None, rate=None):
        super().__init__(var=var, cond_var=cond_var, name=name, features_shape=features_shape,
                         **_valid_param_dict({'concentration': concentration, 'rate': rate}))

    @property
    def params_keys(self):
        return ["concentration", "rate"]

    @property
    def distribution_torch_class(self):
        return GammaTorch

    @property
    def distribution_name(self):
        return "Gamma"

    @property
    def has_reparam(self):
        return True
