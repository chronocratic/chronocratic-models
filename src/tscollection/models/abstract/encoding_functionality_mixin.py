__all__ = ['EncodingFunctionalityMixin']

from collections.abc import Callable
import logging

import torch
from torch import nn
import torch.nn.functional as F  ## noqa: N812
from torch.utils.data import DataLoader, TensorDataset
import tqdm

from src.tscollection.models.encoders.masking import MaskMode
from src.tscollection.models.utils import (
    apply_slicing,
    concat_last_step_features,
    extract_features_from_batch,
    full_series_pooling,
    integer_pooling,
    multiscale_pooling,
    process_sliding_window,
)

logger = logging.getLogger(__name__)


class EncodingFunctionalityMixin:
    _encoder: nn.Module
    _eval_method: Callable[..., torch.Tensor]
    _slice: slice | None = None
    # Declared here so subclass branches can use normal attribute access.
    # CoST provides query_encoder; TS2Vec and AutoTCL provide _averaged_encoder.
    # device is provided by LightningModule.
    query_encoder: nn.Module
    _averaged_encoder: nn.Module
    device: torch.device

    def _pick_the_encoder(self) -> None:
        if hasattr(self, 'query_encoder'):
            self._encoder = self.query_encoder
        else:
            self._encoder = self._averaged_encoder

    def _pick_eval_method(self) -> None:
        if hasattr(self, 'query_encoder'):
            self._eval_method = self._evaluate_with_feature_concatenation
        else:
            self._eval_method = self._evaluate_with_pooling

    def _pick_slice(self, sliding_padding: int, sliding_length: int) -> None:
        if hasattr(self, 'query_encoder'):
            self._slice = None
        else:
            self._slice = slice(sliding_padding, sliding_padding + sliding_length)

    def _evaluate_with_pooling(
        self,
        input_tensor: torch.Tensor,
        mask: torch.Tensor | None = None,
        slicing: slice | None = None,
        encoding_window: str | int | None = None,
    ) -> torch.Tensor:
        """
        Evaluate the model with pooling.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape
            (batch_size, sequence_length, num_features).
            mask (torch.Tensor | None): Optional mask tensor.
            slicing (slice | None): Optional slice to apply on the output.
            encoding_window (str | int | None): Specifies the pooling strategy.

        Returns:
            torch.Tensor: The output tensor after applying the specified pooling strategy.
        """
        output_tensor = self._encoder(
            x=input_tensor.to(self.device, non_blocking=True), mask_mode=mask
        )

        if encoding_window == 'full_series':
            output_tensor = full_series_pooling(tensor=output_tensor, slicing=slicing)
        elif encoding_window == 'multiscale':
            output_tensor = multiscale_pooling(tensor=output_tensor, slicing=slicing)
        elif isinstance(encoding_window, int):
            output_tensor = integer_pooling(
                tensor=output_tensor, encoding_window=encoding_window, slicing=slicing
            )
        else:
            output_tensor = apply_slicing(tensor=output_tensor, slicing=slicing)

        return output_tensor.cpu()

    def _evaluate_with_feature_concatenation(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output_trend_tensor, output_seasonality_tensor = self._encoder(
            x=input_tensor.to(self.device, non_blocking=True), mask_mode=None
        )
        output_tensor = concat_last_step_features(output_trend_tensor, output_seasonality_tensor)
        return output_tensor.cpu()

    def _compute_sliding_representations(
        self,
        input_tensor: torch.Tensor,
        sliding_length: int,
        sliding_padding: int,
        causal: bool,  ## noqa: FBT001
        mask: MaskMode | None,
        encoding_window: str | int | None,
        num_samples: int,
        batch_size: int,
    ) -> torch.Tensor:
        all_representations = []
        calculation_buffer = []
        calculation_buffer_length = 0
        time_series_length = input_tensor.size(1)

        self._pick_slice(sliding_padding, sliding_length)

        # use tqdm to show progress bar
        for start_index in tqdm.tqdm(
            range(0, time_series_length, sliding_length),
            desc='Sliding inference',
            unit='window',
            leave=False,
        ):
            left_index = start_index - sliding_padding
            right_index = start_index + sliding_length + (sliding_padding if not causal else 0)
            sliding_window = process_sliding_window(
                input_tensor=input_tensor,
                left_index=left_index,
                right_index=right_index,
                time_series_length=time_series_length,
            )
            if num_samples < batch_size:
                if calculation_buffer_length + num_samples > batch_size:
                    concatenated_buffer = torch.cat(calculation_buffer, dim=0)
                    representations = self._eval_method(
                        input_tensor=concatenated_buffer,
                        mask=mask,
                        slicing=self._slice,
                        encoding_window=encoding_window,
                    )
                    all_representations += torch.split(representations, num_samples)
                    calculation_buffer = []
                    calculation_buffer_length = 0
                calculation_buffer.append(sliding_window)
                calculation_buffer_length += num_samples

            else:
                representations = self._eval_method(
                    input_tensor=sliding_window,
                    mask=mask,
                    slicing=self._slice,
                    encoding_window=encoding_window,
                )
                all_representations.append(representations)

        if num_samples < batch_size and calculation_buffer_length > 0:
            concatenated_buffer = torch.cat(calculation_buffer, dim=0)
            representations = self._eval_method(
                input_tensor=concatenated_buffer,
                mask=mask,
                slicing=self._slice,
                encoding_window=encoding_window,
            )
            all_representations += torch.split(representations, num_samples)

        concatenated_representations = torch.cat(all_representations, dim=1)

        if encoding_window == 'full_series':
            concatenated_representations = F.max_pool1d(
                concatenated_representations.transpose(1, 2).contiguous(),
                kernel_size=concatenated_representations.size(1),
            ).squeeze(1)
        return concatenated_representations

    def encode(
        self,
        data: torch.Tensor,
        batch_size: int,
        num_workers: int,
        mask: MaskMode | None = None,
        encoding_window: str | int | None = None,
        causal: bool = False,  ## noqa: FBT002 FBT001
        sliding_length: int | None = None,
        sliding_padding: int = 0,
    ) -> torch.Tensor:
        """
        Compute representations using the model.

        Args:
            data (torch.Tensor): This should have a shape of (n_instance, n_timestamps, n_features).
            All missing data should be set to NaN.
            batch_size (int | None): The batch size used for inference.
            num_workers (int): The number of workers used for data loading.
            mask (MaskMode | None): The mask used by the encoder
            can be specified with this parameter.
            This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (str | int | None): When this param is specified,
            the computed representation would be the max pooling over this window.
            This can be set to 'full_series', 'multiscale'
            or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future information
            would not be encoded into the representation of each timestamp.
            sliding_length (int | None): The length of the sliding window.
            When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length
            used for inference every sliding window.


        Returns:
            torch.Tensor: The representations for data.
        """
        self._pick_the_encoder()
        self._pick_eval_method()

        expected_ndim = 3
        if data.ndim != expected_ndim:
            msg = 'Input data must have shape (n_instance, n_timestamps, n_features).'
            raise ValueError(msg)

        num_samples, time_series_length, _ = data.shape  ## noqa: RUF059

        original_training_state = self._encoder.training
        self._encoder.eval()

        dataset = TensorDataset(data)
        msg = f'building data loader with batch size {batch_size} and num workers {num_workers}'
        logging.info(msg)  ## noqa: LOG015
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

        with torch.inference_mode():
            all_outputs = []
            for batch in tqdm.tqdm(
                loader, desc='Encoding data', unit='batch', leave=True, total=len(loader)
            ):
                input_tensor = extract_features_from_batch(batch)

                if sliding_length is not None:
                    representations = self._compute_sliding_representations(
                        input_tensor=input_tensor,
                        sliding_length=sliding_length,
                        sliding_padding=sliding_padding,
                        causal=causal,
                        mask=mask,
                        encoding_window=encoding_window,
                        num_samples=num_samples,
                        batch_size=batch_size,
                    )
                else:
                    representations = self._eval_method(
                        input_tensor=input_tensor, mask=mask, encoding_window=encoding_window
                    )

                    if encoding_window == 'full_series':
                        representations = representations.squeeze(1)

                all_outputs.append(representations.cpu())

            all_outputs = torch.cat(all_outputs, dim=0)

        self._encoder.train(original_training_state)
        return all_outputs
