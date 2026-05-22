import random
from typing import TypedDict, Unpack

import numpy as np
import torch

from .timevae import TimeVAE
from .vae_conv_model import VariationalAutoencoderConv as VAE_Conv
from .vae_dense_model import VariationalAutoencoderDense as VAE_Dense

_VAEModel = VAE_Dense | VAE_Conv | TimeVAE


class _VAEKwargs(TypedDict, total=False):
    latent_dim: int
    reconstruction_wt: float
    learning_rate: float
    hidden_layer_sizes: list | None
    trend_poly: int
    custom_seas: list | None
    use_residual_conn: bool


def set_seeds(seed: int = 111) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def instantiate_vae_model(
    vae_type: str, sequence_length: int, feature_dim: int, **kwargs: Unpack[_VAEKwargs]
) -> _VAEModel:
    set_seeds(seed=123)

    if vae_type == 'vae_dense':
        return VAE_Dense(seq_len=sequence_length, feat_dim=feature_dim, **kwargs)
    if vae_type == 'vae_conv':
        return VAE_Conv(seq_len=sequence_length, feat_dim=feature_dim, **kwargs)
    if vae_type == 'timeVAE':
        return TimeVAE(seq_len=sequence_length, feat_dim=feature_dim, **kwargs)

    msg = f'Unrecognized model type [{vae_type}]. Please choose from vae_dense, vae_conv, timeVAE.'
    raise ValueError(msg)


def get_posterior_samples(vae: _VAEModel, data: np.ndarray) -> np.ndarray:
    return vae.predict(data)


def get_prior_samples(vae: _VAEModel, num_samples: int) -> np.ndarray:
    return vae.get_prior_samples(num_samples=num_samples)
