import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform, Normal, Categorical

from ..utils import log_sum_exp


class GaussianEncoderBase(nn.Module):
    """docstring for EncoderBase"""

    def __init__(self, args):
        super(GaussianEncoderBase, self).__init__()

        self.mixture = bool(args.num_components)
        self.num_components = args.num_components
        self.nz = args.nz

        if self.mixture:
            # Fixed, uniform mixture weights
            mixture_weights = torch.Tensor(self.num_components).fill_(1./self.num_components)
            self.register_buffer("mixture_weights", mixture_weights)

            # Means and variances
            mixture_mu = Uniform(-float(self.num_components), float(self.num_components)
                                 ).sample(torch.Size([self.num_components, 1])).repeat(1, self.nz)
            mixture_var = torch.ones([self.num_components, self.nz], dtype=torch.float)

            if args.learned_prior:
                self.mixture_mu = nn.Parameter(mixture_mu)
                self.mixture_var = nn.Parameter(mixture_var)
            else:
                self.register_buffer("mixture_mu", mixture_mu)
                self.register_buffer("mixture_var", mixture_var)

            cc, ckl = self.check_collapsed_components()
            print('init cc: {}, init ckl: {}'.format(cc, ckl))
        else:
            loc = torch.zeros(self.nz, device=args.device)
            scale = torch.ones(self.nz, device=args.device)

            self.prior = Normal(loc, scale)

    def forward(self, x):
        """
        Args:
            x: (batch_size, *)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        raise NotImplementedError

    def sample(self, input, nsamples):
        """sampling from the encoder
        Returns: Tensor1, Tuple
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tuple: contains the tensor mu [batch, nz] and
                logvar[batch, nz]
        """

        # (batch_size, nz)
        mu, logvar = self.forward(input)

        # (batch, nsamples, nz)
        z = self.reparameterize(mu, logvar, nsamples)

        return z, (mu, logvar)

    def check_collapsed_components(self, threshold=0.1):
        if self.mixture:
            mu, var = self.mixture_mu, F.softplus(self.mixture_var)
            logvar = torch.log(var)
            # (K, K)
            KLs = 0.5 * (logvar.unsqueeze(0) - logvar.unsqueeze(1) + var.unsqueeze(1) / var.unsqueeze(0)
                         + (mu.unsqueeze(0) - mu.unsqueeze(1)) ** 2 / var.unsqueeze(0) - 1).sum(2)
            avg_KL = KLs.mean()
            collapsed_KLs = (KLs < threshold).sum()
            return collapsed_KLs, avg_KL
        else:
            return torch.tensor(0), torch.tensor(0.0)

    def sample_from_prior(self, nsamples):
        if self.mixture:
            # We first sample num_samples modes of the mixture prior, and then sample each mode
            # [num, ]
            k = Categorical(logits=torch.log(self.mixture_weights)).sample(
                torch.Size([nsamples])).long()
            mu, var = self.mixture_mu, F.softplus(self.mixture_var)

            # [num, z_dim]
            mu = torch.index_select(mu, 0, k)
            var = torch.index_select(var, 0, k)
            return torch.normal(mu, torch.sqrt(var))
        else:
            return self.prior.sample((nsamples,))

    def encode(self, input, nsamples):
        """perform the encoding and compute the KL term

        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]

        """

        # (batch_size, nz)
        mu, logvar = self.forward(input)

        # (batch, nsamples, nz)
        z = self.reparameterize(mu, logvar, nsamples)

        if self.mixture:
            # (batch_size, nsamples, nz) X (batch_size, 1, nz) -> (batch_size, nsamples)
            log_q = sample_log_likelihood(z, mu.unsqueeze(1), logvar.exp().unsqueeze(1), 2)
            log_p = self.mixture_log_probability(z)

            # (batch_size)
            KL = (log_q - log_p).mean(dim=1)
        else:
            KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

        return z, KL

    def mixture_log_probability(self, z):
        # (K, nz)
        p_var = F.softplus(self.mixture_var)
        p_mu = self.mixture_mu

        # (batch_size, nsamples, 1, nz) X (K, nz) -> (batch_size, nsamples, K)
        mixture_log = sample_log_likelihood(z.unsqueeze(2), p_mu, p_var, 3)

        # MoG sample probability is the logsumexp of individual mixture components
        log_k = torch.log(self.mixture_weights)
        # (batch_size, n_samples)
        return torch.logsumexp(mixture_log + log_k, dim=2)

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)

            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)

        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)

    def eval_inference_dist(self, x, z, param=None):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        nz = z.size(2)

        if not param:
            mu, logvar = self.forward(x)
        else:
            mu, logvar = param

        # (batch_size, 1, nz)
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        # (batch_size, nsamples, nz)
        dev = z - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density

    def calc_mi(self, x):
        """Approximate the mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))

        Returns: Float

        """

        # [x_batch, nz]
        mu, logvar = self.forward(x)

        x_batch, nz = mu.size()

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).mean()

        # [z_batch, 1, nz]
        z_samples = self.reparameterize(mu, logvar, 1)

        # [1, x_batch, nz]
        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
            0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()


def sample_log_likelihood(z, mu, var, dim):
    return -0.5 * torch.sum((torch.log(2*np.pi*var) + (z - mu)**2 / var), dim=dim)
