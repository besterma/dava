from __future__ import annotations

from torch import Tensor, device, dtype
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List

import torch
from torchtyping import TensorType
import torch.nn as nn
import torch.nn.functional
import numpy as np


def compute_gaussian_kl(z_mean: TensorType["batch", "z_dim"], z_logvar: TensorType["batch", "z_dim"]):
    # z_sigma = torch.exp(z_logvar / 2)
    # therefore "0.5 * (z_sigma ** 2 + z_mean ** 2 - 2 * torch.log(z_sigma) - 1).mean()" becomes:
    return 0.5 * (torch.square(z_mean) + torch.exp(z_logvar) - z_logvar - 1)


def gaussian_log_density(z_sampled: TensorType["batch", "num_latents"],
                         z_mean: TensorType["batch", "num_latents"],
                         z_logvar: TensorType["batch", "num_latents"]):
    normalization = torch.log(torch.tensor(2. * np.pi))
    inv_sigma = torch.exp(-z_logvar)
    tmp = (z_sampled - z_mean)
    return -0.5 * (tmp * tmp * inv_sigma + z_logvar + normalization)


def total_correlation(z: TensorType["batch", "num_latents"],
                      z_mean: TensorType["batch", "num_latents"],
                      z_logvar: TensorType["batch", "num_latents"],
                      dataset_size: int) -> Tensor:
    log_qz_prob = gaussian_log_density(z.unsqueeze(1), z_mean.unsqueeze(0), z_logvar.unsqueeze(0))

    log_qz_product = torch.sum(
        torch.logsumexp(log_qz_prob, dim=1),
        dim=1
    )
    log_qz = torch.logsumexp(
        torch.sum(log_qz_prob, dim=2),
        dim=1
    )
    return torch.mean(log_qz - log_qz_product)


class ConvEncoder(nn.Module):
    def __init__(self, output_dim: int, num_channels: int, use_batch_norm: bool = False,
                 use_spectral_norm: bool = False, use_layer_norm: bool = False,
                 use_instance_norm: bool = False, scale_factor=1):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.use_batch_norm = use_batch_norm
        self.use_spectral_norm = use_spectral_norm
        self.use_layer_norm = use_layer_norm
        self.use_instance_norm = use_instance_norm
        self.scale_factor = scale_factor
        assert not (use_layer_norm and use_batch_norm), "Cant use both layer and batch norm"

        self.conv1 = nn.Conv2d(num_channels, 32*scale_factor, 4, 2, 1)  # 32 x 32
        self.conv1 = nn.utils.spectral_norm(self.conv1) if self.use_spectral_norm else self.conv1
        self.bn1 = nn.BatchNorm2d(32*scale_factor) if self.use_batch_norm else \
                    (nn.LayerNorm([32*scale_factor, 32, 32]) if self.use_layer_norm else
                    (nn.InstanceNorm2d(32*scale_factor) if self.use_instance_norm else None))
        self.conv2 = nn.Conv2d(32*scale_factor, 32*scale_factor, 4, 2, 1)  # 16 x 16
        self.conv2 = nn.utils.spectral_norm(self.conv2) if self.use_spectral_norm else self.conv2
        self.bn2 = nn.BatchNorm2d(32*scale_factor) if self.use_batch_norm else \
                    (nn.LayerNorm([32*scale_factor, 16, 16]) if self.use_layer_norm else
                    (nn.InstanceNorm2d(32*scale_factor) if self.use_instance_norm else None))
        self.conv3 = nn.Conv2d(32*scale_factor, 64*scale_factor, 4, 2, 1)  # 8 x 8
        self.conv3 = nn.utils.spectral_norm(self.conv3) if self.use_spectral_norm else self.conv3
        self.bn3 = nn.BatchNorm2d(64*scale_factor) if self.use_batch_norm else \
                    (nn.LayerNorm([64*scale_factor, 8, 8]) if self.use_layer_norm else
                    (nn.InstanceNorm2d(64*scale_factor) if self.use_instance_norm else None))
        self.conv4 = nn.Conv2d(64*scale_factor, 64*scale_factor, 4, 2, 1)  # 4 x 4
        self.conv4 = nn.utils.spectral_norm(self.conv4) if self.use_spectral_norm else self.conv4
        self.bn4 = nn.BatchNorm2d(64*scale_factor) if self.use_batch_norm else \
                   (nn.LayerNorm([64*scale_factor, 4, 4]) if self.use_layer_norm else
                   (nn.InstanceNorm2d(64*scale_factor) if self.use_instance_norm else None))
        # self.dense1 = nn.Linear(1600, 256) was used in disentanglement_lib code
        self.dense1 = nn.Linear(1024*scale_factor, 256*scale_factor)
        self.dense1 = nn.utils.spectral_norm(self.dense1) if self.use_spectral_norm else self.dense1
        self.bn5 = nn.BatchNorm1d(256*scale_factor) if self.use_batch_norm else \
                   (nn.LayerNorm(256*scale_factor) if self.use_layer_norm else
                   (nn.BatchNorm1d(256*scale_factor) if self.use_instance_norm else None))
        self.dense_means = nn.Linear(256*scale_factor, output_dim)
        self.dense_means = nn.utils.spectral_norm(self.dense_means) if self.use_spectral_norm else self.dense_means
        self.dense_log_var = nn.Linear(256*scale_factor, output_dim)
        self.dense_log_var = nn.utils.spectral_norm(self.dense_log_var) if self.use_spectral_norm else self.dense_log_var

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "output_dim"], TensorType["batch", "output_dim"]):
        h = x.view(-1, self.num_channels, 64, 64)
        h = self.act(self.conv1(h))
        h = self.bn1(h) if self.use_batch_norm or self.use_layer_norm or self.use_instance_norm else h
        h = self.act(self.conv2(h))
        h = self.bn2(h) if self.use_batch_norm or self.use_layer_norm or self.use_instance_norm else h
        h = self.act(self.conv3(h))
        h = self.bn3(h) if self.use_batch_norm or self.use_layer_norm or self.use_instance_norm else h
        h = self.act(self.conv4(h))
        h = self.bn4(h) if self.use_batch_norm or self.use_layer_norm or self.use_instance_norm else h
        # h = h.view(-1, 1600)
        h = h.reshape(-1, 1024*self.scale_factor)
        h = self.act(self.dense1(h))
        h = self.bn5(h) if self.use_batch_norm or self.use_layer_norm or self.use_instance_norm else h
        means = self.dense_means(h)
        log_var = self.dense_log_var(h)

        return means, log_var

    def state_dict(self, destination=None, prefix: str = '', keep_vars: bool = False):
        state_dict = super(ConvEncoder, self).state_dict(destination, prefix, keep_vars)
        state_dict["use_batch_norm"] = self.use_batch_norm
        state_dict["use_layer_norm"] = self.use_layer_norm
        state_dict["use_instance_norm"] = self.use_instance_norm
        return state_dict


class Discriminator(ConvEncoder):
    def __init__(self, output_dim: int, num_channels: int, use_batch_norm: bool = False,
                 use_spectral_norm: bool = False, use_layer_norm: bool = False,
                 use_instance_norm: bool = False, scale_factor: int = 1):
        super(Discriminator, self).__init__(output_dim, num_channels, use_batch_norm,
                                            use_spectral_norm, use_layer_norm, use_instance_norm,
                                            scale_factor)
        self.output_act = nn.Sigmoid()

    def forward(self, x: TensorType["batch", "num_channels", "x", "y"]):
        z, _ = super(Discriminator, self).forward(x)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim: int, num_channels: int, use_batch_norm: bool = False,
                 use_spectral_norm: bool = False, use_layer_norm: bool = False,
                 use_instance_norm: bool = False, scale_factor=1):
        super(ConvDecoder, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.use_batch_norm = use_batch_norm  # https://discuss.pytorch.org/t/autocast-with-normalization-layers/94125
        self.use_spectral_norm = use_spectral_norm
        self.use_layer_norm = use_layer_norm
        self.use_instance_norm = use_instance_norm
        self.scale_factor = scale_factor
        assert not (use_layer_norm and use_batch_norm), "Cant use both layer and batch norm"

        self.dense1 = nn.Linear(input_dim, 256*scale_factor)
        self.dense1 = nn.utils.spectral_norm(self.dense1) if self.use_spectral_norm else self.dense1
        self.bn1 = nn.BatchNorm1d(256*scale_factor) if self.use_batch_norm else \
                    (nn.LayerNorm(256*scale_factor) if self.use_layer_norm else
                    (nn.BatchNorm1d(256*scale_factor) if self.use_instance_norm else None))
        self.dense2 = nn.Linear(256*scale_factor, 1024*scale_factor)
        self.dense2 = nn.utils.spectral_norm(self.dense2) if self.use_spectral_norm else self.dense2
        self.bn2 = nn.BatchNorm2d(64*scale_factor) if self.use_batch_norm else \
                   (nn.LayerNorm([64*scale_factor, 4, 4]) if self.use_layer_norm else
                   (nn.InstanceNorm2d(64*scale_factor) if self.use_instance_norm else None))
        self.conv1 = nn.ConvTranspose2d(64*scale_factor, 64*scale_factor, 4, 2, 1)  # 8 x 8
        self.conv1 = nn.utils.spectral_norm(self.conv1) if self.use_spectral_norm else self.conv1
        self.bn3 = nn.BatchNorm2d(64*scale_factor) if self.use_batch_norm else \
                   (nn.LayerNorm([64*scale_factor, 8, 8]) if self.use_layer_norm else
                   (nn.InstanceNorm2d(64*scale_factor) if self.use_instance_norm else None))
        self.conv2 = nn.ConvTranspose2d(64*scale_factor, 32*scale_factor, 4, 2, 1)  # 16 x 16
        self.conv2 = nn.utils.spectral_norm(self.conv2) if self.use_spectral_norm else self.conv2
        self.bn4 = nn.BatchNorm2d(32*scale_factor) if self.use_batch_norm else \
                   (nn.LayerNorm([32*scale_factor, 16, 16]) if self.use_layer_norm else
                   (nn.InstanceNorm2d(32*scale_factor) if self.use_instance_norm else None))
        self.conv3 = nn.ConvTranspose2d(32*scale_factor, 32, 4, 2, 1)  # 32 x 32
        self.conv3 = nn.utils.spectral_norm(self.conv3) if self.use_spectral_norm else self.conv3
        self.bn5 = nn.BatchNorm2d(32) if self.use_batch_norm else \
                   (nn.LayerNorm([32, 32, 32]) if self.use_layer_norm else
                   (nn.InstanceNorm2d(32) if self.use_instance_norm else None))
        self.conv_final = nn.ConvTranspose2d(32, num_channels, 4, 2, 1)  # 64 x 64
        self.conv_final = nn.utils.spectral_norm(self.conv_final) if self.use_spectral_norm else self.conv_final

        self.act = nn.ReLU(inplace=True)

    def forward(self, z: TensorType["batch", "input_dim"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        h = z.view(-1, self.input_dim)
        h = self.act(self.dense1(h))
        h = self.bn1(h) if self.use_batch_norm or self.use_layer_norm or self.use_instance_norm else h
        h = self.act(self.dense2(h))
        h = h.view(-1, 64*self.scale_factor, 4, 4)
        h = self.bn2(h) if self.use_batch_norm or self.use_layer_norm or self.use_instance_norm else h

        h = self.act(self.conv1(h))
        h = self.bn3(h) if self.use_batch_norm or self.use_layer_norm or self.use_instance_norm else h
        h = self.act(self.conv2(h))
        h = self.bn4(h) if self.use_batch_norm or self.use_layer_norm or self.use_instance_norm else h
        h = self.act(self.conv3(h))
        h = self.bn5(h) if self.use_batch_norm or self.use_layer_norm or self.use_instance_norm else h
        mu_img = self.conv_final(h)

        return mu_img

    def state_dict(self, destination = None, prefix: str = '', keep_vars: bool = False):
        state_dict = super(ConvDecoder, self).state_dict(destination, prefix, keep_vars)
        state_dict["use_batch_norm"] = self.use_batch_norm
        state_dict["use_layer_norm"] = self.use_layer_norm
        state_dict["use_instance_norm"] = self.use_instance_norm
        return state_dict


class FVAEDiscriminator(nn.Module):
    def __init__(self, z_dim):
        super(FVAEDiscriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )

    def forward(self, z):
        return self.net(z).squeeze()


class BetaVAE(nn.Module):
    def __init__(self, z_dim: int, num_channels: int, beta: int, use_batch_norm: bool = False,
                 use_spectral_norm: bool = False, use_layer_norm: bool = False,
                 use_instance_norm: bool = False, scale_factor: int = 1, **kwargs):
        super(BetaVAE, self).__init__(**kwargs)
        assert beta > 0
        assert num_channels > 0
        assert z_dim > 0
        self.z_dim = z_dim
        self.num_channesl = num_channels
        self.beta = beta
        self.encoder = ConvEncoder(z_dim, num_channels, use_batch_norm, use_spectral_norm,
                                   use_layer_norm, use_instance_norm, scale_factor)
        self.decoder = ConvDecoder(z_dim, num_channels, use_batch_norm, use_spectral_norm,
                                   use_layer_norm, use_instance_norm, scale_factor)
        self.N = torch.distributions.Normal(0, 1)
        self.bernoulli_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.tc = 0
        self.kl_loss = 0
        self.reconstruction_loss = 0
        self.latent_loss = 0
        self.combined_loss = 0

    def forward(self, x: TensorType["batch", "num_channels", "x", "y"],
                step: int
                ) -> TensorType:
        z_mean, z_logvar = self.encoder(x)
        z_var = torch.exp(z_logvar / 2)
        z_sampled = z_mean + z_var * self.N.sample(z_mean.shape)
        reconstruction = self.decoder(z_sampled)
        self.kl_loss = compute_gaussian_kl(z_mean, z_logvar).mean()
        self.tc = total_correlation(z_sampled, z_mean, z_logvar, 0)
        # The sigmoid activation gets applied in the BCEWithLogitsLoss
        per_sample_reconstruction_loss = torch.sum(self.bernoulli_loss(reconstruction, x), dim=[1, 2, 3])
        self.reconstruction_loss = torch.mean(per_sample_reconstruction_loss)

        return reconstruction, z_sampled, self.combine_loss(step)

    def combine_loss(self, step: int):
        self.latent_loss = self.beta * self.kl_loss
        self.combined_loss = self.latent_loss + self.reconstruction_loss
        return self.combined_loss

    def reconstruct(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "num_channels", "x", "y"], TensorType["batch", "z_dim"]):
        z_mean, z_logvar = self.encoder(x)
        z_var = torch.exp(z_logvar / 2)
        z_sampled = z_mean + z_var * self.N.sample(z_mean.shape)
        reconstruction = self.decoder(z_sampled).sigmoid()
        return reconstruction, z_sampled

    def reconstruct_deterministic(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "num_channels", "x", "y"], TensorType["batch", "z_dim"]):
        z_mean, z_logvar = self.encoder(x)
        reconstruction = self.decoder(z_mean).sigmoid()
        return reconstruction, z_mean

    def encode(self, x: TensorType["batch", "num_channels", "x", "y"]
                ) -> (TensorType["batch", "output_dim"], TensorType["batch", "output_dim"]):
        return self.encoder(x)

    def decode(self, z: TensorType["batch", "input_dim"]
                ) -> TensorType["batch", "num_channels", "x", "y"]:
        return self.decoder(z).sigmoid()

    def get_enc_layer_weights(self):
        return self.encoder

    @property
    def scale_factor(self):
        return self.encoder.scale_factor

    def to(self: BetaVAE, device: Optional[Union[int, device]] = ..., dtype: Optional[Union[dtype, str]] = ...,
           non_blocking: bool = ...) -> BetaVAE:
        super(BetaVAE, self).to(device)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        return self


class BetaTCVAE(BetaVAE):
    def __init__(self, z_dim: int, num_channels: int, beta: int, use_batch_norm: bool = False,
                 use_spectral_norm: bool = False, use_layer_norm: bool = False,
                 use_instance_norm: bool = False, scale_factor: int = 1, **kwargs):
        super(BetaTCVAE, self).__init__(z_dim, num_channels, beta, use_batch_norm,
                                        use_spectral_norm, use_layer_norm, use_instance_norm, scale_factor, **kwargs)

    def combine_loss(self, step: int):
        self.latent_loss = (self.beta - 1.) * self.tc + self.kl_loss
        self.combined_loss = self.latent_loss + self.reconstruction_loss
        return self.combined_loss


class AnnealedVAE(BetaVAE):
    def __init__(self, z_dim: int, num_channels: int, gamma: int, c_max: float, iteration_threshold: int,
                 annealed_loss_degree=4,
                 use_batch_norm: bool = False, use_spectral_norm: bool = False, use_layer_norm: bool = False,
                 use_instance_norm: bool = False, scale_factor: int = 1, **kwargs):
        super(AnnealedVAE, self).__init__(z_dim, num_channels, 1,
                                          use_batch_norm=use_batch_norm,
                                          use_spectral_norm=use_spectral_norm,
                                          use_layer_norm=use_layer_norm,
                                          use_instance_norm=use_instance_norm,
                                          scale_factor=scale_factor,
                                          **kwargs)
        self.gamma = gamma
        self.c_max = c_max
        self.iteration_threshold = iteration_threshold
        self.loss_degree = annealed_loss_degree

    def anneal(self, step):
        """Anneal function for anneal_vae (https://arxiv.org/abs/1804.03599).

        Args:
          c_max: Maximum capacity.
          step: Current step.
          iteration_threshold: How many iterations to reach c_max.

        Returns:
          Capacity annealed linearly until c_max.
        """
        return np.min((self.c_max, self.c_max * step / self.iteration_threshold))

    def combine_loss(self, step: int):
        if self.loss_degree == 0:
            self.latent_loss = self.gamma * torch.abs(self.kl_loss - self.anneal(step))
        else:
            self.latent_loss = self.gamma * torch.pow(self.kl_loss - self.anneal(step), self.loss_degree)
        self.combined_loss = self.latent_loss + self.reconstruction_loss
        return self.combined_loss

