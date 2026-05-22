__all__ = ['calculate_mutual_information', 'calculate_regular_consistency']

from typing import cast

import numpy as np
import torch

from tscollection.models.augmentation import AugmentationMethod
from tscollection.models.losses import l1_out_loss


def calculate_regular_consistency(weights: torch.Tensor) -> torch.Tensor:
    """
    Calculate regular consistency for weights, i.e., compare differences betw. selected time steps.

    Parameters
    ----------
    weights : torch.Tensor
        The input weight tensor of shape (batch_size, time_steps, channels).

    Returns:
    -------
    torch.Tensor
        The mean consistency measure across the batch.
    """
    batch_size, time_steps, _ = weights.shape

    # Select random time steps for comparison
    selected_steps = torch.randint(1, time_steps - 2, [batch_size])
    left_steps = selected_steps - 1
    right_steps = selected_steps + 1
    other_selected_steps = torch.randint(1, time_steps - 2, [batch_size])

    # Create mask to differentiate between near and far time steps
    mask = torch.where(
        (other_selected_steps - selected_steps) > 1,
        torch.ones_like(other_selected_steps),
        torch.zeros_like(selected_steps),
    ).to(weights.device)

    # Calculate differences for consistency
    differences = (
        mask.view(1, batch_size, 1)
        * torch.abs(weights[:, selected_steps, :] - weights[:, other_selected_steps, :])
        + torch.abs(weights[:, selected_steps, :] - weights[:, left_steps, :])
        + torch.abs(weights[:, selected_steps, :] - weights[:, right_steps, :])
        + (1 - mask).view(1, batch_size, 1)
        * (1 - torch.abs(weights[:, selected_steps, :] - weights[:, other_selected_steps, :]))
    )

    return differences.mean()


def calculate_mutual_information(
    batch: torch.Tensor, augmentation_method: AugmentationMethod, max_train_length: int | None
) -> float:
    """
    Calculate the mutual information (MI) between original and augmented data.

    Parameters
    ----------
    batch: torch.Tensor
        The input batch of data.
    augmentation_method : AugmentationMethod
        The augmentation method to use.
    max_train_length : int
        The maximum length of the training sequences.

    Returns:
    -------
    The average MI between original and augmented data.
    """
    with torch.inference_mode():
        x = batch
        device = x.device

        if max_train_length is not None and x.size(1) > max_train_length:
            window_offset = np.random.randint(x.size(1) - max_train_length + 1)  # noqa: NPY002
            x = x[:, window_offset : window_offset + max_train_length]
        x = x.to(device)

        original_x = x
        augmented_x = augmentation_method.augment(x)

        batch_info_nce_loss = l1_out_loss(original_x, cast('torch.Tensor', augmented_x))
    return batch_info_nce_loss.item()
