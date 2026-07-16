from abc import ABC, abstractmethod

__all__ = ["BaseVariationalAutoencoder", "Sampling"]

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn

from chronocratic.models.utils import (
    extract_features_from_batch,
    masked_reconstruction_loss_sum,
    zero_fill_padding,
)


class Sampling(nn.Module):
    """Reparameterization layer for VAE latent sampling."""

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Sample a latent vector from ``(z_mean, z_log_var)``."""
        z_mean, z_log_var = inputs
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim, device=z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class BaseVariationalAutoencoder(pl.LightningModule, ABC):
    _encoder: nn.Module
    _decoder: nn.Module

    @property
    def encoder(self) -> nn.Module:
        """Return the encoder submodule."""
        if not hasattr(self, "_encoder") or self._encoder is None:
            msg = f"{self.__class__.__name__} must initialize self._encoder"
            raise NotImplementedError(msg)
        return self._encoder

    @property
    def decoder(self) -> nn.Module:
        """Return the decoder submodule."""
        if not hasattr(self, "_decoder") or self._decoder is None:
            msg = f"{self.__class__.__name__} must initialize self._decoder"
            raise NotImplementedError(msg)
        return self._decoder

    def __init__(
        self,
        *,
        sequence_length: int,
        input_dim: int,
        latent_dim: int,
        reconstruction_weight: float = 3.0,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reconstruction_weight = reconstruction_weight
        self.learning_rate = learning_rate
        self.sampling = Sampling()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct an input batch using the latent mean.

        Expects ``x`` of shape ``(batch, sequence_length, input_dim)``.
        The encoder transposes to ``(batch, input_dim, sequence_length)`` internally.
        """
        z_mean, _z_log_var, _z = self._encoder(x)
        return self._decoder(z_mean)

    def _step(
        self, batch: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = extract_features_from_batch(batch)
        x, keep_mask = zero_fill_padding(x)  # (B, T, C), (B, T)
        z_mean, z_log_var, z = self._encoder(x)
        # Original TF always feeds sampled z to decoder in both train and test.
        reconstruction = self._decoder(z)
        loss, recon_loss, kl_loss = self.loss_function(
            x, reconstruction, z_mean, z_log_var, keep_mask=keep_mask
        )
        return loss, recon_loss, kl_loss

    def training_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute, log, and return the training loss for one batch."""
        loss, recon_loss, kl_loss = self._step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_recon_loss", recon_loss, on_epoch=True)
        self.log("train_kl_loss", kl_loss, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, _batch_idx: int) -> torch.Tensor:
        """Compute, log, and return the validation loss for one batch."""
        loss, recon_loss, kl_loss = self._step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_recon_loss", recon_loss, on_epoch=True)
        self.log("val_kl_loss", kl_loss, on_epoch=True)
        return loss

    def configure_optimizers(self):  # noqa: ANN201 (Lightning expects this signature: https://lightning.ai/docs/pytorch/stable/common/optimization.html)
        """Return Adam optimizer with ReduceLROnPlateau scheduler.

        Matches the original TF training pipeline, which always adds
        ``ReduceLROnPlateau(factor=0.5, patience=30)``. The scheduler
        monitors ``train_loss_epoch`` (Lightning's epoch-level aggregated
        metric from ``self.log``) and halves the LR when improvement stalls.

        Additionally, PyTorch Adam uses ``eps=1e-7`` to match Keras Adam
        (default ``epsilon=1e-7``), instead of PyTorch's own default of
        ``eps=1e-8``.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-7)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=30, mode="min"
        )

        lr_scheduler_config_dict = {
            "name": "ReduceLROnPlateau",
            "scheduler": lr_scheduler,
            "monitor": "train_loss_epoch",
            "interval": "epoch",
            "frequency": 1,
            "reduce_on_plateau": True,
            "strict": True,
        }

        return [optimizer], [lr_scheduler_config_dict]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return reconstructions for a NumPy input batch."""
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            x_t = torch.FloatTensor(x).to(next(self.parameters()).device)
            z_mean, _z_log_var, _z = self._encoder(x_t)
            x_decoded = self._decoder(z_mean)
        self.train(was_training)
        return x_decoded.cpu().detach().numpy()

    def get_num_trainable_variables(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_prior_samples(self, num_samples: int) -> np.ndarray:
        """Sample from the standard normal prior and decode the samples."""
        device = next(self.parameters()).device
        with torch.inference_mode():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self._decoder(z)
        return samples.cpu().detach().numpy()

    def get_prior_samples_given_z(self, z: np.ndarray) -> np.ndarray:
        """Decode the provided latent vectors."""
        z_t = torch.as_tensor(z, dtype=torch.float, device=next(self.parameters()).device)
        samples = self._decoder(z_t)
        return samples.cpu().detach().numpy()

    @abstractmethod
    def _build_encoder(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def _build_decoder(self) -> nn.Module:
        raise NotImplementedError

    def _get_reconstruction_loss(
        self, x: torch.Tensor, x_recons: torch.Tensor, keep_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        err = torch.pow(x - x_recons, 2)  # (B, T, C)

        # Per-element reconstruction loss (masked if keep_mask provided)
        if keep_mask is not None:
            reconst_loss = masked_reconstruction_loss_sum(err, keep_mask)
        else:
            reconst_loss = torch.sum(err)

        # Per-axis reconstruction term (mean over channels, sum over B, T)
        # KL and per-axis terms operate over reduced dimensions, not timesteps,
        # so they remain unmasked (consistent with the original implementation).
        x_r = torch.mean(x, dim=2)  # (B, T)
        x_c_r = torch.mean(x_recons, dim=2)  # (B, T)
        axis_err = torch.pow(x_r - x_c_r, 2)  # (B, T)
        reconst_loss += torch.sum(axis_err)

        return reconst_loss

    def loss_function(
        self,
        x: torch.Tensor,
        x_recons: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        *,
        keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total, reconstruction, and KL losses for a batch.

        Args:
            x: Original input, shape ``(B, T, C)``.
            x_recons: Reconstruction, shape ``(B, T, C)``.
            z_mean: Latent mean, shape ``(B, latent_dim)``.
            z_log_var: Latent log-variance, shape ``(B, latent_dim)``.
            keep_mask: Boolean mask of shape ``(B, T)`` where ``True``
                indicates non-padded (real) timesteps. When ``None``,
                all timesteps are included (backward compatible).

        Returns:
            Tuple ``(total_loss, reconstruction_loss, kl_loss)``.
            KL loss is always unmasked (operates over latent dimensions).
        """
        reconstruction_loss = self._get_reconstruction_loss(x, x_recons, keep_mask=keep_mask)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        total_loss = self.reconstruction_weight * reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss
