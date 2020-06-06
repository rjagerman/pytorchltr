"""DCG and NDCG evaluation metrics."""
import torch as _torch
from pytorchltr.utils import rank_by_score as _rank_by_score


def ndcg(scores, relevance, n, k=None, exp=True):
    """Computes Normalized Discounted Cumulative Gain (NDCG).

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
        A tensor of size (batch_size, list_size) indicating the NDCG of each
        query at every rank. If k is not None, then this returns a tensor of
        size (batch_size), indicating the NDCG@k of each query.
    """
    idcg = dcg(relevance.float(), relevance, n, k, exp)
    idcg[idcg == 0.0] = 1.0
    return dcg(scores, relevance, n, k, exp) / idcg


def dcg(scores, relevance, n, k=None, exp=True):
    """Computes Discounted Cumulative Gain (DCG).

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
        A tensor of size (batch_size, list_size) indicating the DCG of each
        query at every rank. If k is not None, then this returns a tensor of
        size (batch_size), indicating the DCG@k of each query.
    """
    # Compute relevance per rank
    rel_sort = _torch.gather(relevance, 1, _rank_by_score(scores, n)).float()
    arange = _torch.repeat_interleave(
        _torch.arange(scores.shape[1], dtype=_torch.float,
                      device=scores.device).reshape(
            (1, scores.shape[1])),
        scores.shape[0], dim=0)
    if exp:
        rel_sort = (2.0 ** rel_sort - 1.0)
    per_rank_dcg = rel_sort / _torch.log2(arange + 2.0)
    dcg = _torch.cumsum(per_rank_dcg, dim=1)

    # Do cutoff at k (or return all dcg@k as an array)
    if k is not None:
        dcg = dcg[:, :k][:, -1]
    return dcg
