"""Collection of PBM-based click simulators."""
from typing import Optional
from typing import Tuple

import torch as _torch
from pytorchltr.utils import mask_padded_values as _mask_padded_values


_SIM_RETURN_TYPE = Tuple[_torch.LongTensor, _torch.FloatTensor]


def simulate_pbm(rankings: _torch.LongTensor, ys: _torch.LongTensor,
                 n: _torch.LongTensor, relevance_probs: _torch.FloatTensor,
                 cutoff: Optional[int] = None,
                 eta: float = 1.0) -> _SIM_RETURN_TYPE:
    """Simulates clicks according to a position-biased user model.

    Args:
        rankings: A tensor of size (batch_size, list_size) of rankings.
        ys: A tensor of size (batch_size, list_size) of relevance labels.
        n: A tensor of size (batch_size) indicating the nr docs per query.
        relevance_prob: A tensor of size (max_relevance) where the entry at
            index "i" indicates the probability of clicking a document with
            relevance label "i" (given that it is observed).
        cutoff: The maximum list size to simulate.
        eta: The severity of position bias (0.0 = no bias)

    Returns:
        A tuple of two tensors of size (batch_size, list_size), where the first
        indicates the clicks with 0.0 and 1.0 and the second indicates the
        propensity of observing each document.
    """
    # Cutoff at n for observation probabilities.
    if cutoff is not None:
        n = _torch.min(_torch.ones_like(n) * cutoff, n)

    # Compute position-biased observation probabilities.
    ranks = 1.0 + _torch.arange(
        rankings.shape[1], device=rankings.device, dtype=_torch.float)
    obs_probs = 1.0 / (1.0 + ranks) ** eta
    obs_probs = _torch.repeat_interleave(
        obs_probs[None, :], rankings.shape[0], dim=0)
    obs_probs = _mask_padded_values(obs_probs, n, mask_value=0.0, mutate=True)

    # Compute relevance labels at every rank.
    ranked_ys = _torch.gather(ys, 1, rankings)

    # Compute click probabilities (given observed).
    relevance_probs = _torch.repeat_interleave(
        relevance_probs[None, :], rankings.shape[0], dim=0)
    click_probs = _torch.gather(relevance_probs, 1, ranked_ys)

    # Sample clicks from bernoulli distribution with probabilities.
    clicks = _torch.bernoulli(click_probs * obs_probs)

    # Invert back to regular ranking.
    invert_ranking = _torch.argsort(rankings, dim=1)

    # Return click realization and propensities.
    return (
        _torch.gather(clicks, 1, invert_ranking).to(dtype=_torch.long),
        _torch.gather(obs_probs, 1, invert_ranking)
    )


def simulate_perfect(rankings: _torch.LongTensor, ys: _torch.LongTensor,
                     n: _torch.LongTensor, cutoff: Optional[int] = None):
    """Simulates clicks according to a perfect user model.

    Args:
        rankings: A tensor of size (batch_size, list_size) of rankings.
        ys: A tensor of size (batch_size, list_size) of relevance labels.
        n: A tensor of size (batch_size) indicating the nr docs per query.
        cutoff: The maximum list size to simulate.

    Returns:
        A tuple of two tensors of size (batch_size, list_size), where the first
        indicates the clicks with 0.0 and 1.0 and the second indicates the
        propensity of observing each document.
    """
    rel_probs = _torch.FloatTensor(
        [0.0, 0.2, 0.4, 0.8, 1.0], device=rankings.device)
    return simulate_pbm(rankings, ys, n, rel_probs, cutoff, 0.0)


def simulate_position(rankings: _torch.LongTensor, ys: _torch.LongTensor,
                      n: _torch.LongTensor, cutoff: Optional[int] = None,
                      eta: float = 1.0) -> _SIM_RETURN_TYPE:
    """Simulates clicks according to a binary position-biased user model.

    Args:
        rankings: A tensor of size (batch_size, list_size) of rankings.
        ys: A tensor of size (batch_size, list_size) of relevance labels.
        n: A tensor of size (batch_size) indicating the nr docs per query.
        cutoff: The maximum list size to simulate.
        eta: The severity of position bias (0.0 = no bias)

    Returns:
        A tuple of two tensors of size (batch_size, list_size), where the first
        indicates the clicks with 0.0 and 1.0 and the second indicates the
        propensity of observing each document.
    """
    rel_probs = _torch.FloatTensor(
        [0.1, 0.1, 0.1, 1.0, 1.0], device=rankings.device)
    return simulate_pbm(rankings, ys, n, rel_probs, cutoff, eta)


def simulate_nearrandom(rankings: _torch.LongTensor, ys: _torch.LongTensor,
                        n: _torch.LongTensor, cutoff: Optional[int] = None,
                        eta: float = 1.0) -> _SIM_RETURN_TYPE:
    """Simulates clicks according to a near-random user model.

    Args:
        rankings: A tensor of size (batch_size, list_size) of rankings.
        ys: A tensor of size (batch_size, list_size) of relevance labels.
        n: A tensor of size (batch_size) indicating the nr docs per query.
        cutoff: The maximum list size to simulate.
        eta: The severity of position bias (0.0 = no bias)

    Returns:
        A tuple of two tensors of size (batch_size, list_size), where the first
        indicates the clicks with 0.0 and 1.0 and the second indicates the
        propensity of observing each document.
    """
    rel_probs = _torch.FloatTensor(
        [0.4, 0.45, 0.5, 0.55, 0.6], device=rankings.device)
    return simulate_pbm(rankings, ys, n, rel_probs, cutoff, eta)
