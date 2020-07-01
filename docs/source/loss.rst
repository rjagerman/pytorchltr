.. _loss:

Loss functions
==============

PyTorchLTR provides serveral common loss functions for LTR. Each loss function
operates on a batch of query-document lists with corresponding relevance
labels.

The input to an LTR loss function comprises three tensors:

  - scores: A tensor of size :math:`(N, \texttt{list_size})`: the item scores
  - relevance: A tensor of size :math:`(N, \texttt{list_size})`: the relevance labels
  - n: A tensor of size :math:`(N)`: the number of docs per learning instance.

And produces the following output:

  - output: A tensor of size :math:`(N)`: the loss per learning instance in the batch.

Example
-------

The following is a usage example for the pairwise hinge loss but the same usage
pattern holds for all the other losses.

.. doctest::

    >>> import torch
    >>> from pytorchltr.loss import PairwiseHingeLoss
    >>> scores = torch.tensor([[0.5, 2.0, 1.0], [0.9, -1.2, 0.0]])
    >>> relevance = torch.tensor([[2, 0, 1], [0, 1, 0]])
    >>> n = torch.tensor([3, 2])
    >>> loss_fn = PairwiseHingeLoss()
    >>> loss_fn(scores, relevance, n)
    tensor([6.0000, 3.1000])
    >>> loss_fn(scores, relevance, n).mean()
    tensor(4.5500)

Additive ranking losses
-----------------------

Additive ranking losses optimize linearly decomposible ranking metrics
:cite:`loss-joachims2002optimizing,loss-agarwal2019general`. These loss
functions optimize an upper bound on the rank of relevant documents via either
a hinge or logistic formulation.

.. autoclass:: pytorchltr.loss.PairwiseHingeLoss
   :members:

   .. automethod:: forward

.. autoclass:: pytorchltr.loss.PairwiseDCGHingeLoss
   :members:

   .. automethod:: forward

.. autoclass:: pytorchltr.loss.PairwiseLogisticLoss
   :members:

   .. automethod:: __init__
   .. automethod:: forward


LambdaLoss
----------

LambdaLoss :cite:`loss-wang2018lambdaloss` is a probabilistic framework for
ranking metric optimization. We provide implementations for ARPLoss1, ARPLoss2,
NDCGLoss1 and NDCGLoss2.

.. autoclass:: pytorchltr.loss.LambdaARPLoss1
   :members:

   .. automethod:: __init__
   .. automethod:: forward

.. autoclass:: pytorchltr.loss.LambdaARPLoss2
   :members:

   .. automethod:: __init__
   .. automethod:: forward

.. autoclass:: pytorchltr.loss.LambdaNDCGLoss1
   :members:

   .. automethod:: __init__
   .. automethod:: forward

.. autoclass:: pytorchltr.loss.LambdaNDCGLoss2
   :members:

   .. automethod:: __init__
   .. automethod:: forward

.. rubric:: References

.. bibliography:: references.bib
    :cited:
    :style: authoryearstyle
    :keyprefix: loss-
