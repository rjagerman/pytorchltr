"""DCG and NDCG evaluation metrics."""
from typing import Optional

import torch as _torch
from pytorchltr.utils import rank_by_score as _rank_by_score


def ndcg(scores: _torch.FloatTensor, relevance: _torch.LongTensor,
         n: _torch.LongTensor, k: Optional[int] = None,
         exp: Optional[bool] = True) -> _torch.FloatTensor:
    r"""Normalized Discounted Cumulative Gain (NDCG)

    .. math::

        \text{ndcg}(\mathbf{s}, \mathbf{y})
        = \frac{\text{dcg}(\mathbf{s}, \mathbf{y})}
        {\text{dcg}(\mathbf{y}, \mathbf{y})}

    Args:
        scores: A tensor of size (batch_size, list_size, 1) or
            (batch_size, list_size), indicating the scores per document per
            query.
        relevance: A tensor of size (batch_size, list_size), indicating the
            relevance judgements per document per query.
        n: A tensor of size (batch_size) indicating the number of docs per
            query.
        k: An integer indicating the cutoff for ndcg.
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


def dcg(scores: _torch.FloatTensor, relevance: _torch.LongTensor,
        n: _torch.LongTensor, k: Optional[int] = None,
        exp: Optional[bool] = True) -> _torch.FloatTensor:
    r"""Discounted Cumulative Gain (DCG)

    .. math::

        \text{dcg}(\mathbf{s}, \mathbf{y})
        = \sum_{i=1}^n \frac{\text{gain}(y_{\pi_i})}{\log_2(1 + i)}

    where :math:`\pi_i` is the index of the item at rank :math:`i` after
    sorting the scores, and:

    .. math::
        :nowrap:

        \[
        \text{gain}(y_i) = \left\{
        \begin{array}{ll}
        2^{y_i} - 1 & \text{if } \texttt{exp=True} \\
        y_i & \text{otherwise}
        \end{array}
        \right.
        \]


    Args:
        scores: A tensor of size (batch_size, list_size, 1) or
            (batch_size, list_size), indicating the scores per document per
            query.
        relevance: A tensor of size (batch_size, list_size), indicating the
            relevance judgements per document per query.
        n: A tensor of size (batch_size) indicating the number of docs per
            query.
        k: An integer indicating the cutoff for ndcg.
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
