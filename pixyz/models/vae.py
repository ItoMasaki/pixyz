import torch
from torch import optim

from ..losses import KullbackLeibler
from ..models.model import Model
from ..utils import tolist


class VAE(Model):
    """
    Variational Autoencoder.

    In VAE class, reconstruction loss on given distributions (encoder and decoder) is set as the default loss class.
    However, if you want to add additional terms, e.g., the KL divergence between encoder and prior,
    you need to set them to the `regularizer` argument, which defaults to None.

    References
    ----------
    [Kingma+ 2013] Auto-Encoding Variational Bayes
    """
    def __init__(self, encoder, decoder, prior=None,
                 optimizer=optim.Adam,
                 optimizer_params={},
                 clip_grad_norm=None,
                 clip_grad_value=None):
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

        # set distributions (for training)
        distributions = [encoder, decoder, prior]

        # set losses
        reconstruction = -decoder.log_prob().expectation(encoder)

        # set regularizer
        regularizer = None if prior is None else KullbackLeibler(encoder, prior)

        loss = (reconstruction + regularizer).mean()

        super().__init__(loss, test_loss=loss,
                         distributions=distributions,
                         optimizer=optimizer, optimizer_params=optimizer_params,
                         clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value)

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

    def train(self, train_x_dict={}, **kwargs):
        return super().train(train_x_dict, **kwargs)

    def test(self, test_x_dict={}, **kwargs):
        return super().test(test_x_dict, **kwargs)

    def check_parameters(self):
        self.batch_size = self.kwargs["batch_size"]
        self.epoch = self.kwargs["epoch"]
        self.latent_dim = self.kwargs["latent_dim"]
        self.KL_param = self.kwargs["KL_param"]

    def update(self, **kwargs):

        # Recieve the message
        data = self.get_observations()
        mu_prior = self.get_backward_msg() # P(z|x)


        # If mu_prior is not calculated yet
        if mu_prior is None:
            mu_prior = torch.zeros(self.batch_size, self.latent_dim)
        else:
            mu_prior = torch.Tensor(np.array(mu_prior))
        self.prior.loc = mu_prior


        # Create a dataset
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(data[0]), torch.from_numpy(data[0]))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)


        # Train the model for self.__epoch times
        for _ in range(self.epoch):
            for x, _ in loader:
                print(x.shape)
                input_dict = {"x": x, "beta": self.KL_param}
                loss = self.train(input_dict, **kwargs)


        # Sampling
        z = self.encoder.sample({"x": torch.Tensor(data[0])})["z"]
        x = self.decoder.sample({"z": z})["x"]
        z = z.detach().cpu().numpy()
        x = x.detach().cpu().numpy()


        # Pass the message
        self.set_forward_msg(z)
        self.send_backward_msgs([x])


        return loss
