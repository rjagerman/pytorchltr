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
        k: (Optional) an integer indicating the cutoff for ndcg.
        exp: A boolean indicating whether to use the exponential notation of
            DCG.

    Returns:
        A tensor of size (batch_size) indicating the ARP of each query.
    """
    # Compute relevance per rank
    rel_sort = _torch.gather(relevance, 1, _rank_by_score(scores, n)).float()
    arange = _torch.repeat_interleave(
        _torch.arange(rel_sort.shape[1])[None, :], rel_sort.shape[0], dim=0)
    return 1.0 + _torch.mean(arange * rel_sort, dim=1)
