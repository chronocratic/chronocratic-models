__all__ = ['calculate_regular_consistency']

import torch


def calculate_regular_consistency(weights: torch.Tensor) -> torch.Tensor:
    """Calculate regular consistency for weights.

    Compares differences between selected time steps.

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
