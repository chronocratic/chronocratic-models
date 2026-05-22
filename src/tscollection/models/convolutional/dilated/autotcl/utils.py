__all__ = ['calculate_mutual_information', 'calculate_regular_consistency']

import torch

from tscollection.models.augmentation.base import AugmentationMethod
from .losses import l1_out_loss


def calculate_regular_consistency(weights: torch.Tensor) -> torch.Tensor:
    """Calculate regular consistency for weights.

    Compares differences between selected time steps.

    Args:
        weights: Input weight tensor of shape (batch_size, time_steps, channels).
            time_steps must be greater than 3.

    Returns:
        Mean consistency measure across the batch.

    Raises:
        ValueError: If time_steps is less than or equal to 3.
    """
    batch_size, time_steps, _ = weights.shape

    if time_steps <= 3:
        msg = f'calculate_regular_consistency requires time_steps > 3, got {time_steps}'
        raise ValueError(msg)

    # Select random time steps for comparison
    selected_steps = torch.randint(1, time_steps - 2, [batch_size], device=weights.device)
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
    batch: torch.Tensor,
    augmentation_method: AugmentationMethod,
    max_train_length: int | None = None,
) -> float:
    """Calculate mutual information between original and augmented data.

    Uses L1-out loss as a proxy for mutual information estimation.

    Args:
        batch: Input batch of shape ``(batch, time, channels)``.
        augmentation_method: Augmentation strategy to apply.
        max_train_length: Optional maximum sequence length. Sequences
            longer than this are truncated randomly.

    Returns:
        Average MI estimate (L1-out loss) between original and
        augmented data.
    """
    import numpy as np

    with torch.inference_mode():
        x = batch
        device = x.device

        if max_train_length is not None and x.size(1) > max_train_length:
            window_offset = np.random.randint(  # noqa: NPY002
                0, x.size(1) - max_train_length + 1
            )
            x = x[:, window_offset : window_offset + max_train_length]
        x = x.to(device)

        views = augmentation_method.augment(x)
        augmented_x = views.views[0]

        mi = l1_out_loss(x, augmented_x)
    return mi.item()
