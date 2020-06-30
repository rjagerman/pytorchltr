"""Average Relevant Position."""
import torch as _torch
from pytorchltr.utils import rank_by_score as _rank_by_score
from pytorchltr.utils import mask_padded_values as _mask_padded_values


def arp(scores, relevance, n):
    r"""Computes Average Relevant Position (ARP).

    .. math::

        \text{arp}(\mathbf{s}, \mathbf{y})
        = \frac{1}{\sum_{i=1}^n y_i} \sum_{i=1}^n y_{\pi_i} \cdot i

    where :math:`\pi_i` is the index of the item at rank :math:`i` after
    sorting the scores.

    Args:
        scores: A tensor of size (batch_size, list_size, 1) or
            (batch_size, list_size), indicating the scores per document per
            query.
        relevance: A tensor of size (batch_size, list_size), indicating the
            relevance judgements per document per query.
        n: A tensor of size (batch_size) indicating the number of docs per
            query.

    Returns:
        A tensor of size (batch_size) indicating the ARP of each query.
    """
    # Compute relevance per rank
    rel_sort = _torch.gather(relevance, 1, _rank_by_score(scores, n)).float()
    arange = 1.0 + _torch.repeat_interleave(
        _torch.arange(
            rel_sort.shape[1], device=rel_sort.device,
            dtype=_torch.float)[None, :],
        rel_sort.shape[0], dim=0)
    _mask_padded_values(rel_sort, n, mask_value=0.0, mutate=True)
    srp = _torch.sum(arange * rel_sort, dim=1)
    nrp = _torch.sum(rel_sort, dim=1)
    nrp[nrp == 0.0] = 1.0
    return srp / nrp
