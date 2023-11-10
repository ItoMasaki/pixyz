import ..distributions as dists
from .model import Model
from ..losses import ELBO
from ..utils import epsilon

import numpy as np

import torch
import torch.nn as nn


class Normal(dists.Normal):
    def __init__(self, input_dim, name="p"):
        super().__init__(var=["x"], name=name)

        self.loc = nn.Parameter(torch.randn(input_dim))
        self.scale = nn.Parameter(torch.randn(input_dim))

    def forward(self):
        return {"loc": self.loc, "scale": self.softplus(self.scale)}

    def softplus(self, x):
        return torch.log(1 + torch.exp(x))


class Categorical(dists.Categorical):
    def __init__(self, latent_dim, name="p"):
        super().__init__(var=["z"], name=name)

        self.probs = nn.Parameter(torch.randn(latent_dim))

    def forward(self):
        return {"probs": self.sigmoid(self.probs)}

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))


class GMM(Model):
    def __init__(self, latent_dim, input_dim, **kwargs):

        distributions = []
        for i in range(latent_dim):
            distributions.append(Normal(input_dim, name="p_%d" %i))

        prior = Categorical(latent_dim, name="p_{prior}")

        self.p = MixtureModel(distributions=distributions, prior=prior)
        self.post = self.p.posterior()

        self.latent_dim = latent_dim

        loss = ELBO(self.p, self.post)

        super().__init__(loss=loss, **kwargs)

    # def train(self, samples_dict):
    #     samples = samples_dict["x"]

    #     # E-step
    #     posterior = self.post.prob().eval(samples_dict)

    #     # M-step
    #     N_k = posterior.sum(dim=1) + epsilon()  # (n_mix,)

    #     # update probs
    #     probs = N_k / N_k.sum()  # (n_mix,)
    #     self.p.prior.probs = probs

    #     # update loc & scale
    #     loc = (posterior[:, None] @ samples[None]).squeeze(1)  # (n_mix, n_dim)
    #     loc /= (N_k[:, None] + epsilon())

    #     cov = (samples[None, :, :] - loc[:, None, :]) ** 2  # Covariances are set to 0.
    #     var = (posterior[:, None, :] @ cov).squeeze(1)  # (n_mix, n_dim)
    #     var /= (N_k[:, None])
    #     scale = var.sqrt() + epsilon()

    #     for i, d in enumerate(self.p.distributions):
    #         d.loc[0] = loc[i]
    #         d.scale[0] = scale[i]

    #     loss = self.p.log_prob().mean().eval(samples_dict).mean()

    #     return loss

    def test(self, samples_dict):
        samples = samples_dict["x"]

        loss = self.p.log_prob().eval(samples_dict).mean()

        return loss

    def update(self):
        data = self.get_observations()
        Pdz = self.get_backward_msg() # P(z|d)

        N = len(data[0])  # データ数

        # backward messageがまだ計算されていないときは一様分布にする
        if Pdz is None:
            Pdz = np.ones((N, self.latent_dim))/self.latent_dim

        # Create a dataset
        dataset = torch.utils.data.TensorDataset(torch.Tensor(data[0]), torch.Tensor(data[0]))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.kwargs["batch_size"], shuffle=True)

        # if self.__load_dir is None:
        #     save_dir = os.path.join( self.get_name(), "%03d" % self.__n )
        # else:
        #     save_dir = os.path.join( self.get_name(), "recog" )

        # GMM学習
        for _ in range(self.kwargs["epoch"]):
            for x, _ in loader:
                input_dict = {"x": x}
                loss = self.train(input_dict)
                print(loss)

        mu = [self.p.distributions[i].loc[0].detach().numpy() for i in range(len(self.p.distributions))]

        Pdz = self.post.prob().eval(input_dict).detach().numpy()
        print(Pdz.shape)
        Pdz = (Pdz.T / np.sum(Pdz, 1)).T
        
        # self.__n += 1

        # メッセージの送信
        self.set_forward_msg(Pdz)
        self.send_backward_msgs([mu])
