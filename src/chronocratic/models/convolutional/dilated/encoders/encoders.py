__all__ = [
    "AutoTCLAugmentationTimeSeriesEncoder",
    "AutoTCLTimeSeriesEncoder",
    "CoSTTimeSeriesEncoder",
    "TS2VecTimeSeriesEncoder",
]

from abc import ABC, abstractmethod
from collections.abc import Iterable
import itertools

from einops import rearrange, reduce
import torch
from torch import nn

from chronocratic.models.convolutional.dilated.encoders.masking import (
    generate_mask,
    generate_not_nan_mask,
    MaskMode,
)
from chronocratic.models.convolutional.dilated.layers import Conv1dDilatedEncoder
from chronocratic.models.layers import BandedFourierLayer


class BaseTimeSeriesEncoder(nn.Module, ABC):
    """
    Parameters

    input_dims: Number of input dimensions.
    output_dims: Number of output dimensions.
    hidden_dims: Number of hidden dimensions.
    feature_extractor_depth: the depth
    of the feature extractor (the number of convolutional layers).
    dropout_rate: the dropout rate.
    conv_kernel_size: the size of the kernel for the convolutions.
    mask_mode: the mode of masking to use.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: int = 64,
        feature_extractor_depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
    ) -> None:
        super().__init__()

        self.mask_mode = mask_mode

        self.input_fc_layer = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = Conv1dDilatedEncoder(
            in_channels=hidden_dims,
            channels=[hidden_dims] * feature_extractor_depth + [output_dims],
            kernel_size=conv_kernel_size,
        )
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    @abstractmethod
    def _process_not_nan_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def _apply_mask(
        self, x: torch.Tensor, not_nan_mask: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        *,
        return_tcn_output: bool = False,
        mask_mode: MaskMode = MaskMode.ALL_TRUE,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pass

    def _process_mask_mode(self, mask_mode: MaskMode | None) -> MaskMode:
        if mask_mode is None:
            mask_mode = self.mask_mode if self.training else MaskMode.ALL_TRUE
        return mask_mode

    def _common_forward(self, x: torch.Tensor, mask_mode: MaskMode | None = None) -> torch.Tensor:
        # x: BatchSize x TimeSteps x InputDim
        x, not_nan_mask = self._process_not_nan_mask(x=x)

        x = self.input_fc_layer(x)  # BatchSize x TimeSteps x Channels

        mask_mode = self._process_mask_mode(mask_mode=mask_mode)

        mask = generate_mask(x=x, mask_mode=mask_mode)
        x = self._apply_mask(x=x, not_nan_mask=not_nan_mask, mask=mask)

        x = x.transpose(1, 2)  # BatchSize x Channels x TimeSteps
        x = self.feature_extractor(x)  # BatchSize x OutputChannels x TimeSteps

        return x


# -------- AutoTCL Time Series Encoders --------


class AutoTCLTimeSeriesEncoder(BaseTimeSeriesEncoder):
    """
    A class to encode time-series data using a Dilated Convolutional Neural Network based on the implementation of the AutoTCL paper: https://github.com/AslanDing/AutoTCL.

    Parameters
    ----------
    input_dims: Number of input dimensions.
    output_dims: Number of output dimensions.
    kernel_sizes: Tuple of kernel sizes for the convolutions.
    hidden_dims: Number of hidden dimensions.
    feature_extractor_depth: the depth of the feature extractor (the number of convolutional layers).
    dropout_rate: the dropout rate.
    conv_kernel_size: the size of the kernel for the convolutions.
    mask_mode: the mode of masking to use.
    """  # noqa: E501

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        kernel_sizes: tuple[int, ...],
        hidden_dims: int = 64,
        feature_extractor_depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            feature_extractor_depth=feature_extractor_depth,
            dropout_rate=dropout_rate,
            conv_kernel_size=conv_kernel_size,
            mask_mode=mask_mode,
        )

        self.kernel_sizes = kernel_sizes
        self.temporal_feature_decoders = nn.ModuleList(
            [nn.Conv1d(output_dims, output_dims, k, padding=k - 1) for k in kernel_sizes]
        )

    def _process_not_nan_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        not_nan_mask = generate_not_nan_mask(x=x)
        nan_mask_as_float = not_nan_mask.float()
        modified_x = x.clone()
        modified_x = modified_x * nan_mask_as_float.unsqueeze(2)

        return modified_x, not_nan_mask

    def _apply_mask(
        self, x: torch.Tensor, not_nan_mask: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask = mask.float()
        nan_mask_as_float = not_nan_mask.float()
        mask = mask * nan_mask_as_float
        modified_x = x.clone()
        modified_x = modified_x * mask.unsqueeze(2)
        return modified_x

    def _decode_trend(self, x: torch.Tensor) -> torch.Tensor:
        trend = []
        for idx, mod in enumerate(self.temporal_feature_decoders):
            out = mod(x)  # BatchSize x Channels x TimeSteps
            if self.kernel_sizes[idx] != 1:
                out = out[..., : -(self.kernel_sizes[idx] - 1)]  # Adjust for kernel size if not 1
            trend.append(out.transpose(1, 2))  # BatchSize x TimeSteps x Channels

        trend = reduce(
            rearrange(
                trend, "list BatchSize TimeSteps Channels -> list BatchSize TimeSteps Channels"
            ),
            "list BatchSize TimeSteps Channels -> BatchSize TimeSteps Channels",
            "mean",
        )
        return trend

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_tcn_output: bool = False,
        mask_mode: MaskMode = MaskMode.ALL_TRUE,
    ) -> torch.Tensor:
        """Encode a batch of time series into trend representations.

        Args:
            x: Input tensor of shape ``(batch, time, channels)``.
            return_tcn_output: If ``True``, return the raw TCN output
                ``(batch, time, output_dims)`` before trend decoding.
            mask_mode: Masking strategy applied to the input.

        Returns:
            Trend tensor of shape ``(batch, time, output_dims)``, or the raw
            TCN output of the same shape when ``return_tcn_output`` is ``True``.
        """
        x = self._common_forward(x=x, mask_mode=mask_mode)

        if return_tcn_output:
            return x.transpose(1, 2)  # BatchSize x TimeSteps x OutputChannels

        trend = self._decode_trend(x)

        return trend


class AutoTCLAugmentationTimeSeriesEncoder(nn.Module):
    """Augmentation encoder for AutoTCL that learns a differentiable channel mask.

    Wraps an ``AutoTCLTimeSeriesEncoder`` to produce a stochastic binary mask
    over input channels via Gumbel-sigmoid sampling. The mask is applied to the
    raw input, yielding an augmented view used for contrastive training.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        kernel_sizes: tuple[int, ...],
        hidden_dims: int = 64,
        feature_extractor_depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
        num_augmentation_channels: int = 1,
        gumbel_bias: float = 0.001,
        zeta: float = 1.0,
        gamma_zeta: float = 0.05,
        *,
        hard_mask: bool = True,
    ) -> None:

        super().__init__()

        self.gumbel_bias = gumbel_bias
        self.zeta = zeta
        self.gamma_zeta = -gamma_zeta
        self.hard_mask = hard_mask

        self.augmentation_network = AutoTCLTimeSeriesEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            kernel_sizes=kernel_sizes,
            hidden_dims=hidden_dims,
            feature_extractor_depth=feature_extractor_depth,
            dropout_rate=dropout_rate,
            conv_kernel_size=conv_kernel_size,
            mask_mode=mask_mode,
        )

        self.factor_augmentation_network = nn.Sequential(
            nn.Linear(output_dims, num_augmentation_channels), nn.Sigmoid()
        )
        self.augmentation_projector = nn.Sequential(
            nn.Linear(output_dims, output_dims),
            nn.ReLU(),
            nn.Linear(output_dims, num_augmentation_channels),
            nn.Sigmoid(),
        )

    def _sample_graph(
        self,
        sampling_weights: torch.Tensor,
        temperature: float = 1.0,
        bias: float = 0.0,
        *,
        is_training: bool = True,
    ) -> torch.Tensor:
        """
        Obtain a sample graph while maintaining the possibility to backprop.

        Parameters
        ----------
        sampling_weights : torch.Tensor
            Weights provided by the MLP.
        temperature : float, optional
            Annealing temperature to make the procedure more deterministic (default is 1.0).
        bias : float, optional
            Bias on the weights to make sampling less deterministic (default is 0.0).
        is_training : bool, optional
            If set to False, the sampling will be entirely deterministic (default is True).

        Returns:
        -------
        torch.Tensor
            Sample graph.
        """
        if is_training:
            bias = bias + self.gumbel_bias
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()).to(
                sampling_weights.device
            ) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)

        stretched_values = graph * (self.zeta - self.gamma_zeta) + self.gamma_zeta
        clipped = torch.clip(stretched_values, max=1.0, min=0.0)

        return clipped

    def get_parameters(self) -> Iterable[nn.Parameter]:
        """Return all trainable parameters across the three sub-networks.

        Returns:
            Chained iterator over parameters of ``augmentation_network``,
            ``factor_augmentation_network``, and ``augmentation_projector``.
        """
        return itertools.chain(
            self.augmentation_network.parameters(),
            self.factor_augmentation_network.parameters(),
            self.augmentation_projector.parameters(),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_tcn_output: bool = False,
        mask_mode: MaskMode = MaskMode.ALL_TRUE,
    ) -> dict[str, torch.Tensor]:
        """Encode the input and produce an augmented view via a learned channel mask.

        Args:
            x: Input tensor of shape ``(batch, time, channels)``.
            return_tcn_output: If ``True``, skip augmentation and return only
                ``{'embeddings': trend}`` with the raw TCN output.
            mask_mode: Masking strategy applied inside the augmentation network.

        Returns:
            A dict with keys:

            - ``'embeddings'``: trend tensor from the augmentation network.
            - ``'augmented_data'``: ``x`` after applying the learned channel mask.
            - ``'augmentation_factor'``: raw sigmoid scores per channel.
            - ``'projection_factor'``: projected scores from the stop-gradient branch.

            When ``return_tcn_output`` is ``True``, only ``'embeddings'`` is present.
        """
        trend = self.augmentation_network.forward(
            x=x, return_tcn_output=return_tcn_output, mask_mode=mask_mode
        )

        if return_tcn_output:
            return {"embeddings": trend}

        augmentation_factor = self.factor_augmentation_network(trend)

        trend_ = trend.clone()
        projection_factor = self.augmentation_projector(trend_.detach())

        augmentation_mask = self._sample_graph(
            sampling_weights=augmentation_factor, is_training=self.training
        )

        if self.hard_mask:
            hard_mask_h = (torch.sign(augmentation_mask - 0.5) + 1) / 2
            augmentation_mask = (hard_mask_h - augmentation_mask).detach() + augmentation_mask

        augmented_x = projection_factor * augmentation_mask * x

        return {
            "embeddings": trend,
            "augmented_data": augmented_x,
            "augmentation_factor": augmentation_factor,
            "projection_factor": projection_factor,
        }

    def augment(self, data: torch.Tensor) -> torch.Tensor:
        """Return an augmented view of ``data`` using the learned channel mask.

        Args:
            data: Input tensor of shape ``(batch, time, channels)``.

        Returns:
            Augmented tensor of the same shape as ``data``.
        """
        model_output = self.forward(data)

        augmented_x = model_output["augmented_data"]

        return augmented_x

    def get_features(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the full forward pass and return all intermediate outputs.

        Args:
            data: Input tensor of shape ``(batch, time, channels)``.

        Returns:
            Dict with keys ``'embeddings'``, ``'augmented_data'``,
            ``'augmentation_factor'``, and ``'projection_factor'``.
            See :meth:`forward` for details.
        """
        model_output = self.forward(data)

        return model_output


# -------- End AutoTCL Time Series Encoders --------

# -------- CoST Time Series Encoders --------


class CoSTTimeSeriesEncoder(BaseTimeSeriesEncoder):
    """
    A class to encode time-series data using a Convolutional Sparse Transformer implemented based on the CoST paper: https://github.com/salesforce/CoST.

    Parameters
    input_dims : int
        Number of input dimensions.
    output_dims : int
        Number of output dimensions.
    kernel_sizes : tuple[int, ...]
        List of kernel sizes for the convolutions.
    length : int
        Length of the input sequence.
    hidden_dims : int, optional
        Number of hidden dimensions (default is 64).
    feature_extractor_depth : int, optional
        Depth of the feature extractor (number of convolutional layers, default is 10).
    dropout_rate : float, optional
        Dropout rate (default is 0.1).
    conv_kernel_size : int, optional
        Size of the kernel for the convolutions (default is 3).
    mask_mode : MaskMode, optional
        Mode of masking to use (default is MaskMode.BINOMIAL).
    num_bands : int, optional
        Number of bands for the Banded Fourier Layer (default is 1).
    """  # noqa: E501

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        kernel_sizes: tuple[int, ...],
        length: int,
        hidden_dims: int = 64,
        feature_extractor_depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
        num_bands: int = 1,
    ) -> None:
        if output_dims % 2 != 0:
            msg = f"output_dims must be even for CoST, got {output_dims}"
            raise ValueError(msg)

        super().__init__(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            feature_extractor_depth=feature_extractor_depth,
            dropout_rate=dropout_rate,
            conv_kernel_size=conv_kernel_size,
            mask_mode=mask_mode,
        )

        self.kernel_sizes = kernel_sizes

        self.component_dims = output_dims // 2

        self.temporal_feature_decoders = nn.ModuleList(
            [nn.Conv1d(output_dims, output_dims // 2, k, padding=k - 1) for k in kernel_sizes]
        )

        self.seasonal_feature_decoders = nn.ModuleList(
            [
                BandedFourierLayer(
                    in_channels=output_dims,
                    out_channels=self.component_dims,
                    band=b,
                    num_bands=num_bands,
                    length=length,
                )
                for b in range(num_bands)
            ]
        )

    def _process_not_nan_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        not_nan_mask = generate_not_nan_mask(x=x)
        modified_x = x.clone()
        modified_x[~not_nan_mask] = 0
        return modified_x, not_nan_mask

    def _apply_mask(
        self, x: torch.Tensor, not_nan_mask: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask &= not_nan_mask
        modified_x = x.clone()
        modified_x[~mask] = 0
        return modified_x

    def _decode_trend(self, x: torch.Tensor) -> torch.Tensor:
        trend = []
        for idx, mod in enumerate(self.temporal_feature_decoders):
            out = mod(x)  # BatchSize x Channels x TimeSteps
            if self.kernel_sizes[idx] != 1:
                out = out[..., : -(self.kernel_sizes[idx] - 1)]  # Adjust for kernel size if not 1
            trend.append(out.transpose(1, 2))  # BatchSize x TimeSteps x Channels

        trend = reduce(
            rearrange(
                trend, "list BatchSize TimeSteps Channels -> list BatchSize TimeSteps Channels"
            ),
            "list BatchSize TimeSteps Channels -> BatchSize TimeSteps Channels",
            "mean",
        )
        return trend

    def _decode_season(self, x: torch.Tensor) -> list[torch.Tensor]:
        season = []
        for mod in self.seasonal_feature_decoders:
            out = mod(x)  # BatchSize x TimeSteps x Channels
            season.append(out)

        return season

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_tcn_output: bool = False,
        mask_mode: MaskMode = MaskMode.ALL_TRUE,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of time series into disentangled trend and seasonal representations.

        Args:
            x: Input tensor of shape ``(batch, time, channels)``.
            return_tcn_output: If ``True``, return the raw TCN output
                ``(batch, time, output_dims)`` before component decoding.
            mask_mode: Masking strategy applied to the input.

        Returns:
            A 2-tuple ``(trend, season)`` where both tensors have shape
            ``(batch, time, component_dims)``, or the raw TCN output of shape
            ``(batch, time, output_dims)`` when ``return_tcn_output`` is ``True``.
        """
        x = self._common_forward(x=x, mask_mode=mask_mode)

        if return_tcn_output:
            return x.transpose(1, 2)

        trend = self._decode_trend(x)

        x = x.transpose(1, 2)  # BatchSize x TimeSteps x OutputChannels

        season = self._decode_season(x)

        season = season[0]

        return trend, self.dropout_layer(season)


# -------- End CoST Time Series Encoders --------

# -------- TS2Vec Time Series Encoders --------


class TS2VecTimeSeriesEncoder(BaseTimeSeriesEncoder):
    """
    A class to encode time-series data using a Convolutional Sparse Transformer based on the implementation of the TS2Vec paper: https://github.com/zhihanyue/ts2vec.

    Parameters
    input_dims: Number of input dimensions.
    output_dims: Number of output dimensions.
    hidden_dims: Number of hidden dimensions.
    feature_extractor_depth: the depth of the feature extractor (the number of convolutional layers).
    dropout_rate: the dropout rate.
    conv_kernel_size: the size of the kernel for the convolutions.
    mask_mode: the mode of masking to use.
    """  # noqa: E501

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: int = 64,
        feature_extractor_depth: int = 10,
        dropout_rate: float = 0.1,
        conv_kernel_size: int = 3,
        mask_mode: MaskMode = MaskMode.BINOMIAL,
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            feature_extractor_depth=feature_extractor_depth,
            dropout_rate=dropout_rate,
            conv_kernel_size=conv_kernel_size,
            mask_mode=mask_mode,
        )

    def _process_not_nan_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        not_nan_mask = generate_not_nan_mask(x=x)
        modified_x = x.clone()
        modified_x[~not_nan_mask] = 0
        return modified_x, not_nan_mask

    def _apply_mask(
        self, x: torch.Tensor, not_nan_mask: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask &= not_nan_mask
        modified_x = x.clone()
        modified_x[~mask] = 0
        return modified_x

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_tcn_output: bool = False,  # noqa: ARG002
        mask_mode: MaskMode = MaskMode.ALL_TRUE,
    ) -> torch.Tensor:
        """Encode a batch of time series into contextual representations.

        Args:
            x: Input tensor of shape ``(batch, time, channels)``.
            return_tcn_output: Unused; present for interface compatibility.
            mask_mode: Masking strategy applied to the input.

        Returns:
            Encoded tensor of shape ``(batch, time, output_dims)``.
        """
        x = self._common_forward(x=x, mask_mode=mask_mode)

        x = self.dropout_layer(x)
        x = x.transpose(1, 2)  # BatchSize x TimeSteps x OutputChannels

        return x

    # -------- End TS2Vec Time Series Encoders --------
