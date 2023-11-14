
from ..models.gmm import GMM as _GMM

import numpy as np

import torch


class GMM(_GMM):
    def __init__(self, mixture_model, approximate_posterior,
                 epoch=10, latent_dim=10,
                 **kwargs):
        """
        Parameters
        ----------
        mixture_model : MixtureModel
            Mixture model.
        approximate_posterior : Distribution
            Approximate posterior.
        epoch : int
            Number of epochs for training.
        latent_dim : int
            Dimension of latent variable.
        """

        ################
        # Distribution #
        ################
        self.p = mixture_model
        self.post = self.p.posterior()
        self.appx_post = approximate_posterior

        ##############
        # Parameters #
        ##############
        self.epoch = epoch
        self.latent_dim = latent_dim

        super().__init__(self.p, self.appx_post, **kwargs)
        super().setup_serket()

    def update(self):
        data = self.get_observations()
        Pdz = self.get_backward_msg() # P(z|d)


        N = len(data[0])  # データ数


        # backward messageがまだ計算されていないときは一様分布にする
        if Pdz is None:
            Pdz = np.ones((N, self.latent_dim), dtype=np.float32)/self.latent_dim

        
        self.p.prior.probs = torch.nn.Parameter(torch.from_numpy(Pdz))

        
        # GMM学習
        for _ in range(self.epoch):
            loss = self.train({"x": torch.Tensor(data[0])})


        # Passing the message
        Pdz = self.post.prob().eval({"x": torch.Tensor(data[0])}).detach().cpu().numpy() # P(z|d)
        Pdz = (Pdz / np.sum(Pdz, 0))
        mu = [self.p.distributions[idx].loc.detach().cpu().numpy() for idx in np.argmax(Pdz, 0)] # P(d|z)
        

        # メッセージの送信
        self.set_forward_msg(Pdz)
        self.send_backward_msgs([mu])


        return loss
