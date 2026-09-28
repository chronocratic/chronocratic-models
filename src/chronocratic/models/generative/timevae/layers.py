"""TimeVAE-specific decoder layers."""

from collections.abc import Sequence
import warnings

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from chronocratic.models.generative.timevae.enums import ResidualProjectionType

__all__ = ["ResidualConnection"]

# Above this many params, DENSE's final_dense is big enough to be a training-memory risk
# (weights + gradients + Adam state ≈ 16 bytes/param). 50M params ≈ 800 MB for this layer
# alone. Chosen as a round order-of-magnitude threshold, not a hard cutoff — CROP is the
# zero-parameter alternative for anyone who hits it.
_DENSE_PARAM_WARN_THRESHOLD = 50_000_000


class ResidualConnection(nn.Module):
    """Residual decoder branch: latent vector -> ``(B, T, C)`` via ConvTranspose1d stack.

    Two ways to reach exactly ``sequence_length`` steps from the deconvolution output
    (which is always >= ``sequence_length``, see the memory-fix spec §2.7):

    ``ResidualProjectionType.DENSE`` (default): upstream TimeVAE's ``Linear(C * L, C * T)``
    after a ReLU'd deconvolution output. Matches the original implementation. Costs
    O((C * T)^2) parameters — ~676 M at T=5200, C=5 (~10.8 GB of training memory); large
    shapes can exhaust memory.

    ``ResidualProjectionType.CROP``: crop the first ``T`` steps. The last deconvolution is
    linear (no ReLU) so residuals can be negative. No extra parameters. Use this instead
    of DENSE when the upstream-parity cost is too high.
    """

    def __init__(
        self,
        *,
        sequence_length: int,
        input_dim: int,
        hidden_layer_sizes: Sequence[int],
        latent_dim: int,
        encoder_last_dense_dim: int,
        projection: ResidualProjectionType,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.projection = projection

        self.dense = nn.Linear(latent_dim, encoder_last_dense_dim)
        self.deconv_layers: nn.ModuleList = nn.ModuleList()
        in_channels = hidden_layer_sizes[-1]

        for num_filters in reversed(hidden_layer_sizes[:-1]):
            self.deconv_layers.append(
                nn.ConvTranspose1d(
                    in_channels, num_filters, kernel_size=3, stride=2, padding=1, output_padding=1
                )
            )
            in_channels = num_filters

        self.deconv_layers.append(
            nn.ConvTranspose1d(
                in_channels, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1
            )
        )

        length_in = encoder_last_dense_dim // hidden_layer_sizes[-1]
        for _ in range(len(hidden_layer_sizes)):
            length_in = (length_in - 1) * 2 - 2 * 1 + 3 + 1
        length_final = length_in

        if projection is ResidualProjectionType.DENSE:
            final_dense_params = (input_dim * length_final) * (sequence_length * input_dim)
            if final_dense_params > _DENSE_PARAM_WARN_THRESHOLD:
                msg = (
                    f"ResidualConnection: residual_projection=DENSE allocates "
                    f"{final_dense_params:,} params ({final_dense_params * 4 / 1e9:.1f} GB just "
                    f"for weights) at sequence_length={sequence_length}, input_dim={input_dim}. "
                    "Pass residual_projection=ResidualProjectionType.CROP for a zero-parameter "
                    "alternative if this exhausts memory."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
            self.final_dense = nn.Linear(input_dim * length_final, sequence_length * input_dim)
        elif length_final < sequence_length:
            msg = (
                f"ResidualConnection: deconvolution output length {length_final} is shorter "
                f"than sequence_length {sequence_length}; cannot crop."
            )
            raise ValueError(msg)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return the residual decoder branch for each latent vector."""
        batch_size = z.size(0)
        x = F.relu(self.dense(z))
        x = x.view(batch_size, -1, self.hidden_layer_sizes[-1])
        x = x.transpose(1, 2)

        for deconv in list(self.deconv_layers)[:-1]:
            x = F.relu(deconv(x))
        x = self.deconv_layers[-1](x)  # (B, C, L_final), L_final >= T

        if self.projection is ResidualProjectionType.CROP:
            # Linear last layer: residuals must be able to be negative. Transpose, not view:
            # the data is channels-first.
            return x[:, :, : self.sequence_length].transpose(1, 2)  # (B, T, C)

        x = F.relu(x).flatten(1)
        x = self.final_dense(x)
        return x.view(-1, self.sequence_length, self.input_dim)
