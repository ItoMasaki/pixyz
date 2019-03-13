import torch
from torch import nn

from ..utils import get_dict_values
from .losses import Loss


class SimilarityLoss(Loss):
    """
    Learning Modality-Invariant Representations
    for Speech and Images (Leidai et. al.)
    """
    def __init__(self, p1, p2, input_var=None, var=["z"], margin=0):
        super().__init__(p1, p2, input_var)
        self.var = var
        self.loss = nn.MarginRankingLoss(margin=margin, reduce=False)

    def _sim(self, x1, x2):
        return torch.sum(x1*x2, dim=1)

    def _get_estimated_value(self, x={}, **kwargs):

        inputs = get_dict_values(x, self._p1.input_var, True)
        sample1 = get_dict_values(self._p1.sample(inputs), self.var)[0]

        inputs = get_dict_values(x, self._p2.input_var, True)
        sample2 = get_dict_values(self._p2.sample(inputs), self.var)[0]

        batch_size = sample1.shape[0]
        shuffle_id = torch.randperm(batch_size)
        _sample1 = sample1[shuffle_id]
        _sample2 = sample2[shuffle_id]

        sim12 = self._sim(sample1, sample2)
        sim1_2 = self._sim(sample1, _sample2)
        sim_12 = self._sim(_sample1, sample2)

        dummy_label = torch.ones_like(sim12)
        loss = self.loss(sim12, sim1_2, dummy_label) \
            + self.loss(sim12, sim_12, dummy_label)

        # TODO: fix
        sample1.update(sample2)

        return loss, sample1


class MultiModalContrastivenessLoss(Loss):
    """
    Disentangling by Partitioning:
    A Representation Learning Framework for Multimodal Sensory Data
    """
    def __init__(self, p1, p2, input_var=None, margin=0.5):
        super().__init__(p1, p2, input_var)
        self.loss = nn.MarginRankingLoss(margin=margin)

    def _sim(self, x1, x2):
        return torch.exp(-torch.norm(x1-x2, 2, dim=1) / 2)

    def _get_estimated_value(self, x={}, **kwargs):
        inputs = get_dict_values(x, self._p1.input_var, True)
        sample1 = self._p1.sample_mean(inputs)

        inputs = get_dict_values(x, self._p2.input_var, True)
        sample2 = self._p2.sample_mean(inputs)

        batch_size = sample1.shape[0]
        shuffle_id = torch.randperm(batch_size)
        _sample1 = sample1[shuffle_id]
        _sample2 = sample2[shuffle_id]

        sim12 = self._sim(sample1, sample2)
        sim1_2 = self._sim(sample1, _sample2)
        sim_12 = self._sim(_sample1, sample2)

        dummy_label = torch.ones_like(sim12)
        loss = self.loss(sim12, sim1_2, dummy_label) \
            + self.loss(sim12, sim_12, dummy_label)

        # TODO: fix
        return loss, x
