__all__ = ['calculate_regular_consistency']

import torch


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
        msg = (
            f'calculate_regular_consistency requires time_steps > 3, '
            f'got {time_steps}'
        )
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
