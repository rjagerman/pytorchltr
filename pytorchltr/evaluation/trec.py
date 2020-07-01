"""Generate pytrec_eval runs from model and labels."""
from typing import Optional
from typing import Tuple
from typing import Dict
import torch as _torch


_PYTREC_RETURN_TYPE = Tuple[
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, float]]]


def generate_pytrec_eval(scores: _torch.FloatTensor,
                         relevance: _torch.LongTensor,
                         n: _torch.LongTensor,
                         qids: Optional[_torch.LongTensor] = None,
                         qid_offset: int = 0,
                         q_prefix: str = "q",
                         d_prefix: str = "d") -> _PYTREC_RETURN_TYPE:
    """Generates `pytrec_eval <https://github.com/cvangysel/pytrec_eval>`_
    qrels and runs from given batch.

    Example usage:

    .. doctest::

        >>> import json
        >>> import torch
        >>> import pytrec_eval
        >>> from pytorchltr.evaluation.trec import generate_pytrec_eval
        >>> scores = torch.tensor([[1.0, 0.0, 1.5], [1.5, 0.2, 0.5]])
        >>> relevance = torch.tensor([[0, 1, 0], [0, 1, 1]])
        >>> n = torch.tensor([3, 3])
        >>> qrel, run = generate_pytrec_eval(scores, relevance, n)
        >>> evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg'})
        >>> print(json.dumps(evaluator.evaluate(run), indent=1))
        {
         "q0": {
          "map": 0.3333333333333333,
          "ndcg": 0.5
         },
         "q1": {
          "map": 0.5833333333333333,
          "ndcg": 0.6934264036172708
         }
        }

    Args:
        scores: A FloatTensor of size (batch_size, list_size) indicating the
            scores of each document.
        relevance: A LongTensor of size (batch_size, list_size) indicating the
            relevance of each document.
        n: A LongTensor of size (batch_size) indicating the number of docs per
            query.
        qids: A LongTensor of size (batch_size) indicating the qid of each
            query.
        qid_offset: An offset to increment all qids in this batch with. Only
            used if `qids` is None.
        q_prefix: A string prefix to add for query identifiers.
        d_prefix: A string prefix to add for doc identifiers.

    Returns:
        A tuple containing a qrel dict and a run dict.
    """
    qrel = {}
    run = {}
    for i in range(scores.shape[0]):

        # Store under correct qid.
        if qids is not None:
            qid = f"{q_prefix}{int(qids[i])}"
        else:
            qid = f"{q_prefix}{i + qid_offset}"
        qrel[qid] = {}
        run[qid] = {}

        # Iterate documents and get relevance and scores.
        for d in range(n[i]):
            qrel[qid][f"{d_prefix}{d}"] = int(relevance[i, d])
            run[qid][f"{d_prefix}{d}"] = float(scores[i, d])

    return qrel, run
