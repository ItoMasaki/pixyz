from torch import optim, nn
import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import re

from ..model import Model


class Module(Model):
    __counter = 0

    def __init__(self, name="", learnable=True)
        """Set up Serket.

        Parameters
        ----------
        name : str
            Name of Serket
        learnable : bool
            If True, the parameters of Serket are optimized.

        """
        self.__name = "module%03d_" % self.__counter + name
        self.__counter += 1
        self.__forward_prob = None
        self.__backward_prob = None
        self.__learnable = learnable
        self.__observations = None
