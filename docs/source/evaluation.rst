Evaluation
==========

PyTorchLTR provides several built-in evaluation metrics including ARP
:cite:`evaluation-joachims2017unbiased` and DCG
:cite:`evaluation-kalervo2002cumulated`. Furthermore, the library has support
for creating `pytrec_eval <https://github.com/cvangysel/pytrec_eval>`_ 
:cite:`evaluation-gysel2018pytreceval` compatible output.

Example
-------

.. doctest::

    >>> import torch
    >>> from pytorchltr.evaluation import ndcg
    >>> scores = torch.tensor([[1.0, 0.0, 1.5], [1.5, 0.2, 0.5]])
    >>> relevance = torch.tensor([[0, 1, 0], [0, 1, 1]])
    >>> n = torch.tensor([3, 3])
    >>> ndcg(scores, relevance, n, k=10)
    tensor([0.5000, 0.6934])

Built-in metrics
----------------

.. autofunction:: pytorchltr.evaluation.arp

.. autofunction:: pytorchltr.evaluation.dcg

.. autofunction:: pytorchltr.evaluation.ndcg

Integration with pytrec_eval
----------------------------

.. autofunction:: pytorchltr.evaluation.generate_pytrec_eval

.. rubric:: References

.. bibliography:: references.bib
    :cited:
    :style: authoryearstyle
    :keyprefix: evaluation-
