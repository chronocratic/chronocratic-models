from abc import ABC, abstractmethod

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn


class Sampling(nn.Module):
    """Reparameterization layer for VAE latent sampling."""

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Sample a latent vector from ``(z_mean, z_log_var)``."""
        z_mean, z_log_var = inputs
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class BaseVariationalAutoencoder(pl.LightningModule, ABC):
    encoder: nn.Module
    decoder: nn.Module

    def __init__(
        self,
        seq_len: int,
        feat_dim: int,
        latent_dim: int,
        reconstruction_wt: float = 3.0,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.learning_rate = learning_rate
        self.sampling = Sampling()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct an input batch using the latent mean."""
        z_mean, _z_log_var, _z = self.encoder(x)
        return self.decoder(z_mean)

    def _step(
        self, batch: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        loss, recon_loss, kl_loss = self.loss_function(x, reconstruction, z_mean, z_log_var)
        n = x.size(0)
        return loss / n, recon_loss / n, kl_loss / n

    def training_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute, log, and return the training loss for one batch."""
        loss, recon_loss, kl_loss = self._step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', recon_loss, on_epoch=True)
        self.log('train_kl_loss', kl_loss, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute, log, and return the validation loss for one batch."""
        loss, recon_loss, kl_loss = self._step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_epoch=True)
        self.log('val_kl_loss', kl_loss, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return the Adam optimizer used to train the VAE."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return reconstructions for a NumPy input batch."""
        self.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x).to(next(self.parameters()).device)
            z_mean, _z_log_var, _z = self.encoder(x_t)
            x_decoded = self.decoder(z_mean)
        return x_decoded.cpu().detach().numpy()

    def get_num_trainable_variables(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_prior_samples(self, num_samples: int) -> np.ndarray:
        """Sample from the standard normal prior and decode the samples."""
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z)
        return samples.cpu().detach().numpy()

    def get_prior_samples_given_z(self, z: np.ndarray) -> np.ndarray:
        """Decode the provided latent vectors."""
        z_t = torch.FloatTensor(z).to(next(self.parameters()).device)
        samples = self.decoder(z_t)
        return samples.cpu().detach().numpy()

    @abstractmethod
    def _build_encoder(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def _build_decoder(self) -> nn.Module:
        raise NotImplementedError

    def _get_reconstruction_loss(self, x: torch.Tensor, x_recons: torch.Tensor) -> torch.Tensor:
        def get_reconst_loss_by_axis(
            x: torch.Tensor, x_recons: torch.Tensor, dim: int
        ) -> torch.Tensor:
            x_r = torch.mean(x, dim=dim)
            x_c_r = torch.mean(x_recons, dim=dim)
            err = torch.pow(x_r - x_c_r, 2)
            return torch.sum(err)

        err = torch.pow(x - x_recons, 2)
        reconst_loss = torch.sum(err)
        reconst_loss += get_reconst_loss_by_axis(x, x_recons, dim=2)  # by time axis
        return reconst_loss

    def loss_function(
        self, x: torch.Tensor, x_recons: torch.Tensor, z_mean: torch.Tensor, z_log_var: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total, reconstruction, and KL losses for a batch."""
        reconstruction_loss = self._get_reconstruction_loss(x, x_recons)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss
