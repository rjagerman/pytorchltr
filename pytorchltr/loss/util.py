import torch as _torch


def batch_pairwise_difference(x):
    """Returns a pairwise difference matrix.

    This matrix contains entries p_{i,j} = x_i - x_j

    Args:
        x: The input batch of dimension (batch_size, list_size, 1).

    Returns:
        A tensor of size (batch_size, list_size, list_size) containing pairwise
        differences.
    """

    # Construct broadcasted x_{:,i,0...list_size}
    x_ij = _torch.repeat_interleave(x, x.shape[1], dim=2)

    # Construct broadcasted x_{:,0...list_size,i}
    x_ji = _torch.repeat_interleave(x.permute(0, 2, 1), x.shape[1], dim=1)

    # Compute difference
    return x_ij - x_ji
