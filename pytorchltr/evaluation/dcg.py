"""DCG and NDCG evaluation metrics."""
import torch as _torch


def ndcg(scores, relevance, n, k=None, exp=True):
    r"""Computes Normalized Discounted Cumulative Gain (NDCG).

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
    r"""Computes Discounted Cumulative Gain (DCG).

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
    # Compute nr_doc mask to mask out padded documents
    if scores.dim() == 3:
        scores = scores.reshape((scores.shape[0], scores.shape[1]))
    mask = _torch.repeat_interleave(
        _torch.arange(scores.shape[1]).reshape((1, scores.shape[1])),
        scores.shape[0], dim=0)
    n_mask = _torch.repeat_interleave(
        n.reshape((n.shape[0], 1)), scores.shape[1], dim=1)
    scores[mask >= n_mask] = -float('inf')
    relevance[mask >= n_mask] = 0.0

    # Compute relevance per rank
    rel_sort = _torch.gather(relevance, 1, _batched_tiebreak_argsort(scores)).float()
    arange = _torch.repeat_interleave(
        _torch.arange(scores.shape[1], dtype=_torch.float).reshape((1, scores.shape[1])),
        scores.shape[0], dim=0)
    per_rank_dcg = (2.0 ** rel_sort - 1.0) / _torch.log2(arange + 2.0)
    dcg = _torch.cumsum(per_rank_dcg, dim=1)

    # Do cutoff at k (or return all dcg@k as an array)
    if k is not None:
        dcg = dcg[:, :k][:, -1]
    return dcg


def _batched_tiebreak_argsort(x):
    """Computes a per-row argsort of matrix x with random tiebreaks.

    Arguments:
        x: A 2D tensor where each row will be argsorted.

    Returns:
        A 2D tensor of the same size as x, where each row is the argsort of x,
        with ties broken randomly.
    """
    p = _torch.randperm(x.shape[1])
    return p[_torch.argsort(x[:, p], descending=True)]
