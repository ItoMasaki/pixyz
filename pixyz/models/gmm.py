import torch.optim as optim

from ..losses import ELBO
from ..models.model import Model
from ..utils import tolist



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
