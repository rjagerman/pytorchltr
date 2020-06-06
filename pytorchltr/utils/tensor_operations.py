"""Common utils for the library."""
import torch as _torch


def mask_padded_values(xs, n, mask_value=-float('inf'), mutate=False):
    """Turns padded values into given mask value.

    Arguments:
        xs: A tensor of size (batch_size, list_size, 1) containing padded
            values.
        n: A tensor of size (batch_size) containing list size of each query.
        mask_value: The value to mask with (default: -inf).
        mutate: Whether to mutate the values of xs or return a copy.
    """
    mask = _torch.repeat_interleave(
        _torch.arange(xs.shape[1], device=xs.device).reshape((1, xs.shape[1])),
        xs.shape[0], dim=0)
    n_mask = _torch.repeat_interleave(
        n.reshape((n.shape[0], 1)), xs.shape[1], dim=1)
    if not mutate:
        xs = xs.clone()
    xs[mask >= n_mask] = mask_value
    return xs


def tiebreak_argsort(x):
    """Computes a per-row argsort of matrix x with random tiebreaks.

    Arguments:
        x: A 2D tensor where each row will be argsorted.

    Returns:
        A 2D tensor of the same size as x, where each row is the argsort of x,
        with ties broken randomly.
    """
    p = _torch.randperm(x.shape[1], device=x.device)
    return p[_torch.argsort(x[:, p], descending=True)]


def rank_by_score(scores, n):
    """Sorts scores in decreasing order.

    This method ensures that padded documents are placed last and ties are
    broken randomly.

    Arguments:
        scores: A tensor of size (batch_size, list_size, 1) or
                (batch_size, list_size) containing scores.
        n: A tensor of size (batch_size) containing list size of each query.
    """
    if scores.dim() == 3:
        scores = scores.reshape((scores.shape[0], scores.shape[1]))
    return tiebreak_argsort(mask_padded_values(scores, n))


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
