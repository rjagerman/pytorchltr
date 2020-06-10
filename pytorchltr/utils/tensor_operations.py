"""Common utils for the library."""
import torch as _torch


def mask_padded_values(xs, n, mask_value=-float('inf'), mutate=False):
    """Turns padded values into given mask value.

    Args:
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


def tiebreak_argsort(x, descending=True):
    """Computes a per-row argsort of matrix x with random tiebreaks.

    Args:
        x: A 2D tensor where each row will be argsorted.
        descending: Whether to sort in descending order.

    Returns:
        A 2D tensor of the same size as x, where each row is the argsort of x,
        with ties broken randomly.
    """
    p = _torch.randperm(x.shape[1], device=x.device)
    return p[_torch.argsort(x[:, p], descending=descending)]


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


def rank_by_plackettluce(scores, n):
    """Samples a ranking from a plackett luce distribution.

    This method ensures that padded documents are placed last.

    Args:
        scores: A tensor of size (batch_size, list_size, 1) or
                (batch_size, list_size) containing scores.
        n: A tensor of size (batch_size) containing list size of each query.
    """
    if scores.dim() == 3:
        scores = scores.reshape((scores.shape[0], scores.shape[1]))
    masked_scores = mask_padded_values(scores, n)

    # This implementation uses reservoir sampling, which comes down to doing
    # Uniform(0, 1) ^ (1 / p) and then sorting by the resulting values. The
    # following implementation is a numerically stable variant that operates in
    # log-space.
    log_p = _torch.nn.LogSoftmax(dim=1)(masked_scores)
    c_zero = _torch.tensor(0.0).to(device=scores.device)
    c_one = _torch.tensor(1.0).to(device=scores.device)
    u = _torch.distributions.uniform.Uniform(c_zero, c_one).sample(log_p.shape)
    r = _torch.log(-_torch.log(u)) - log_p
    return tiebreak_argsort(r, descending=False)


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
