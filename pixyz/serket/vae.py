import numpy as np

import torch
from torch import optim

from ..models.vae import VAE as _VAE


class VAE(_VAE):
    def __init__(self, encoder, decoder, prior=None,
                 batch_size=128, epoch=100, latent_dim=10, KL_param=1., **kwargs):

        """
        Parameters
        ----------
        encoder : torch.distributions.Distribution
            Encoder distribution.
        decoder : torch.distributions.Distribution
            Decoder distribution.
        prior : torch.distributions.Distribution
            Prior distribution.
        optimizer : torch.optim
            Optimization algorithm.
        optimizer_params : dict
            Parameters of optimizer
        clip_grad_norm : float or int
            Maximum allowed norm of the gradients.
        clip_grad_value : float or int
            Maximum allowed value of the gradients.
        """

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

        self.batch_size = batch_size
        self.epoch = epoch
        self.latent_dim = latent_dim
        self.KL_param = KL_param

        super().__init__(encoder, decoder, prior, **kwargs)
        super().setup_serket()

    def update(self, **kwargs):

        # Recieve the message
        data = self.get_observations() # x
        mu_prior = self.get_backward_msg() # P(z|x)

        N = len(data[0])

        # If mu_prior is not calculated yet
        if mu_prior is None:
            mu_prior = torch.zeros(N, self.latent_dim)
        else:
            mu_prior = torch.from_numpy(np.array(mu_prior)).squeeze()

        # Create a dataset
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(data[0]), mu_prior)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


        # Train the model for self.__epoch times
        total_loss = 0.
        for _ in range(self.epoch):
            for x, prior in loader:
                self.prior.loc = prior
                input_dict = {"x": x, "beta": self.KL_param}
                loss = self.train(input_dict, **kwargs)
                total_loss += loss


        # Sampling
        z = self.encoder.sample({"x": torch.Tensor(data[0])})["z"]
        x = self.decoder.sample({"z": z})["x"]
        z = z.detach().cpu().numpy()
        x = x.detach().cpu().numpy()


        # Pass the message
        self.set_forward_msg(z) # P(z|x)
        self.send_backward_msgs([x]) # P(x|z)


        return total_loss/(self.epoch*len(loader))
