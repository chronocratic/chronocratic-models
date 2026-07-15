import warnings

import torch
from torch import nn
from torch.nn import init

from chronocratic.models.enums.layers import NormalizationLayerType


class DisjoinEncoder(nn.Module):
    """DisjoinEncoder with a selectable normalization strategy and auto-clamp.

    The encoder applies three convolutions in sequence:

    1. **temporal_CNN** (Conv2d): kernel ``(1, temporal_kernel_size)``,
       reduces time dimension by ``temporal_kernel_size - 1``.
    2. **spatial_CNN** (Conv2d): kernel ``(spatial_kernel_size, 1)``,
       reduces channel dimension by ``spatial_kernel_size - 1``.
    3. **rep_CNN** (Conv1d): kernel ``representation_kernel_size``,
       reduces time dimension further by ``representation_kernel_size - 1``.

    When ``sequence_length`` is provided and is shorter than the minimum safe
    length (``temporal_kernel_size + representation_kernel_size - 1``), the
    temporal kernel is automatically clamped and a :class:`UserWarning` is
    emitted.

    Args:
        input_dim: Number of input channels (spatial conv kernel height).
        embedding_dim: Channel count of the temporal/spatial conv blocks.
        representation_dim: Channel count of the representation conv block.
            This is the **per-branch** dimension (each DisjoinEncoder produces
            this many features); the final concatenated output is
            ``2 * representation_dim``.
        temporal_kernel_size: Kernel width of the temporal 2D convolution.
        spatial_kernel_size: Kernel height of the spatial 2D convolution.
            Defaults to ``None``, which resolves to ``input_dim``.
            Must satisfy ``input_dim >= spatial_kernel_size``.
        representation_kernel_size: Kernel width of the 1D representation
            convolution. Defaults to 3.
        sequence_length: Input sequence length. When provided and too short
            for the kernel chain, ``temporal_kernel_size`` is clamped.
        normalization_layer_type: Normalization strategy. ``CHANNEL``
            (default) uses GroupNorm(num_groups=1, C), which is per-sample
            and works correctly at batch_size=1. ``BATCH`` uses
            BatchNorm2d/BatchNorm1d to reproduce the upstream Series2Vec
            architecture exactly.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        embedding_dim: int,
        representation_dim: int,
        temporal_kernel_size: int,
        spatial_kernel_size: int | None = None,
        representation_kernel_size: int = 3,
        sequence_length: int | None = None,
        normalization_layer_type: NormalizationLayerType = NormalizationLayerType.CHANNEL,
    ) -> None:
        super().__init__()

        # Resolve spatial_kernel_size: None means "use input_dim" (D-03)
        resolved_spatial = input_dim if spatial_kernel_size is None else spatial_kernel_size
        if input_dim < resolved_spatial:
            msg = f"input_dim ({input_dim}) must be >= spatial_kernel_size ({resolved_spatial})"
            raise ValueError(msg)

        # Auto-clamp temporal_kernel_size for short sequences (D-02)
        self.temporal_kernel_size = temporal_kernel_size
        if sequence_length is not None:
            min_safe_t = temporal_kernel_size + representation_kernel_size - 1
            if sequence_length < min_safe_t:
                clamped = max(1, sequence_length - representation_kernel_size + 1)
                warnings.warn(
                    f"DisjoinEncoder: sequence_length ({sequence_length}) is shorter than "
                    f"the minimum safe length ({min_safe_t}) for "
                    f"temporal_kernel_size={temporal_kernel_size} and "
                    f"representation_kernel_size={representation_kernel_size}. "
                    f"Clamping temporal_kernel_size from {temporal_kernel_size} to {clamped}.",
                    UserWarning,
                    stacklevel=2,
                )
                self.temporal_kernel_size = clamped

        _temporal_norm = (
            nn.GroupNorm(num_groups=1, num_channels=embedding_dim)
            if normalization_layer_type == NormalizationLayerType.CHANNEL
            else nn.BatchNorm2d(embedding_dim)
        )
        _spatial_norm = (
            nn.GroupNorm(num_groups=1, num_channels=embedding_dim)
            if normalization_layer_type == NormalizationLayerType.CHANNEL
            else nn.BatchNorm2d(embedding_dim)
        )
        _rep_norm = (
            nn.GroupNorm(num_groups=1, num_channels=representation_dim)
            if normalization_layer_type == NormalizationLayerType.CHANNEL
            else nn.BatchNorm1d(representation_dim)
        )

        self.temporal_CNN = nn.Sequential(
            nn.Conv2d(
                1, embedding_dim, kernel_size=(1, self.temporal_kernel_size), padding="valid"
            ),
            _temporal_norm,
            nn.GELU(),
        )

        self.spatial_CNN = nn.Sequential(
            nn.Conv2d(
                embedding_dim, embedding_dim, kernel_size=(resolved_spatial, 1), padding="valid"
            ),
            _spatial_norm,
            nn.GELU(),
        )

        self.rep_CNN = nn.Sequential(
            nn.Conv1d(embedding_dim, representation_dim, kernel_size=representation_kernel_size),
            _rep_norm,
            nn.GELU(),
        )
        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input shaped ``(batch, channels, time)``."""
        x = x.unsqueeze(1)
        x = self.temporal_CNN(x)
        x = self.spatial_CNN(x)
        x = self.rep_CNN(x.squeeze(2))
        return x

    def initialize_weights(self) -> None:
        """Initialize convolution weights with Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.constant_(m.bias, 0)
