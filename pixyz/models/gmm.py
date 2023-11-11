from .model import Model
from ..losses import ELBO
from ..utils import epsilon
from ..distributions import Normal as _Normal, Categorical as _Categorical, MixtureModel

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class GMM(Model):
    def __init__(self, mixture_model, approximate_posterior,
                 optimizer=optim.Adam,
                 optimizer_params={},
                 clip_grad_norm=None,
                 clip_grad_value=None):
        """
        Parameters
        ----------
        mixture_model : MixtureModel
            Mixture model.
        approximate_posterior : Distribution
            Approximate posterior.
        optimizer : torch.optim
            Optimization algorithm.
        optimizer_params : dict
            Parameters of optimizer
        clip_grad_norm : float
            Maximum allowed norm of the gradients.
        clip_grad_value : float
            Maximum allowed value of the gradients.
        """

        self.p = mixture_model
        self.post = self.p.posterior()
        self.appx_post = approximate_posterior
        distributions = [self.p, self.appx_post]

        loss = -ELBO(self.p, self.appx_post).mean()

        super().__init__(loss=loss, test_loss=loss,
                         distributions=distributions,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value,
                         retain_graph=True)

    def train(self, train_x_dict={}, **kwargs):
        return super().train(train_x_dict, **kwargs)

    def test(self, test_x_dict={}, **kwargs):
        return super().test(test_x_dict, **kwargs)

    def check_parameters(self):
        self.batch_size = self.kwargs["batch_size"]
        self.epoch      = self.kwargs["epoch"]
        self.latent_dim = self.kwargs["latent_dim"]

    def update(self):
        data = self.get_observations()
        Pdz = self.get_backward_msg() # P(z|d)

        N = len(data[0])  # データ数

        # backward messageがまだ計算されていないときは一様分布にする
        if Pdz is None:
            Pdz = np.ones((N, self.latent_dim))/self.latent_dim

        # Create a dataset
        dataset = torch.utils.data.TensorDataset(torch.Tensor(data[0]), torch.Tensor(data[0]))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # GMM学習
        for _ in range(self.epoch):
            # for x, _ in loader:
            input_dict = {"x": torch.Tensor(data[0])}
            loss = self.train(input_dict)

        # Passing the message
        Pdz = self.post.prob().eval({"x": torch.Tensor(data[0])}).detach().cpu().numpy() # P(z|d)
        Pdz = (Pdz / np.sum(Pdz, 0))
        mu = [self.p.distributions[idx].loc.detach().cpu().numpy() for idx in np.argmax(Pdz, 0)] # P(d|z)
        

        # メッセージの送信
        self.set_forward_msg(Pdz)
        self.send_backward_msgs([mu])

        return loss
