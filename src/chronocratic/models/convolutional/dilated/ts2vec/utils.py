__all__ = ["extract_subsequences_per_row"]

import torch


def extract_subsequences_per_row(
    array: torch.Tensor, indices: torch.Tensor, num_elements: int
) -> torch.Tensor:
    """
    Extract subsequences from each row of a 2D tensor
    based on provided starting indices and subsequence length.

    Args:
        array (torch.Tensor): The input 2D tensor from which to extract subsequences.
        indices (torch.Tensor): The starting indices for each row, as a 1D tensor.
        num_elements (int): The number of elements to extract from each row.

    Returns:
        torch.Tensor: A tensor containing the extracted subsequences.
    """  ## noqa: D205
    all_indices = indices[:, None] + torch.arange(num_elements)
    # device-ok: CPU indexing
    return array[torch.arange(all_indices.shape[0])[:, None], all_indices]
