"""Average Relevant Position."""
import torch as _torch
from pytorchltr.util import rank_by_score as _rank_by_score


def arp(scores, relevance, n):
    r"""Computes Average Relevant Position (ARP).

    Arguments:
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
    srp = _torch.sum(arange * rel_sort, dim=1)
    nrp = _torch.sum(rel_sort, dim=1)
    nrp[nrp == 0.0] = 1.0
    return srp / nrp
