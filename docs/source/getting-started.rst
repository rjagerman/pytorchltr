Getting started guide
=====================

This guide describes step-by-step instructions for running a full example of
training a neural network on a LTR task.

Data loading
------------

.. warning::

    PyTorchLTR provides utilities to automatically download and prepare several
    public LTR datasets. We do not host or distribute these datasets and it is
    ultimately **your responsibility** to determine whether you have permission
    to use each dataset under its respective license.

The first step is loading the dataset. For this guide we will use the
MSLR-WEB10K dataset which is a learning to rank dataset containing 10,000
queries split across a training (60%), validation (20%) and test (20%) split.
We will use the first fold of the dataset. The following code will
automatically download the dataset if it has not yet been downloaded.
Downloading and processing the data can take a few minutes.

.. code-block:: python

    >>> from pytorchltr.datasets import MSLR10K
    >>> train = MSLR10K(split="train", fold=1)
    >>> test = MSLR10K(split="test", fold=1)

A complete list of available datasets is available :ref:`here <datasets>`.

Building a scoring function
---------------------------

Next, we will set up a scoring function. For this guide we will use a simple
feedforward neural network with ReLU activation functions. This network will,
given a query-document feature vector, predict a score as output.

.. code-block:: python

    >>> import torch
    >>> class Model(torch.nn.Module):
    >>>   def __init__(self, in_features):
    >>>     super().__init__()
    >>>     self.l1 = torch.nn.Linear(in_features, 50)
    >>>     self.l2 = torch.nn.Linear(50, 10)
    >>>     self.l3 = torch.nn.Linear(10, 1)
    >>>   def forward(self, x):
    >>>     o1 = torch.nn.functional.relu(self.l1(x))
    >>>     o2 = torch.nn.functional.relu(self.l2(o1))
    >>>     return self.l3(o2)

With our model class defined, we can now create an instance. We can use the
train dataset to extract the dimensionality of the input feature vectors and
create a model instance.

.. code-block:: python

    >>> torch.manual_seed(42)
    >>> dimensionality = train[0].features.shape[1]
    >>> model = Model(dimensionality)

Training the model
------------------

Next, we will train the model using a basic training loop. First we set up the
loss function and optimizer. For this example we will use a simple
pairwise hinge loss. More information about the available loss functions can be
found :ref:`here <loss>`.

.. code-block:: python

    >>> from pytorchltr.loss import PairwiseHingeLoss
    >>> optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)
    >>> loss_fn = PairwiseHingeLoss()

Next, we will implement the actual training loop which will run for 20 epochs.
We use a collate function with a maximum list size of 20 to truncate the list
of each training instance to a maximum of 20 items. This will significantly
speed up the training process for pairwise losses which have a computational
complexity that is quadratic in the list size.

.. code-block:: python

    >>> from pytorchltr.datasets.list_sampler import UniformSampler
    >>> for epoch in range(1, 21):
    >>>   loader = torch.utils.data.DataLoader(
    >>>     train, batch_size=16, shuffle=True,
    >>>     collate_fn=train.collate_fn(UniformSampler(max_list_size=20)))
    >>>   for batch in loader:
    >>>     xs, ys, n = batch.features, batch.relevance, batch.n
    >>>     loss = loss_fn(model(xs), ys, n).mean()
    >>>     optimizer.zero_grad()
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>   print("Finished epoch %d" % epoch)
    Finished epoch 1
    Finished epoch 2
    Finished epoch 3
    Finished epoch 4
    Finished epoch 5
    Finished epoch 6
    Finished epoch 7
    Finished epoch 8
    Finished epoch 9
    Finished epoch 10
    Finished epoch 11
    Finished epoch 12
    Finished epoch 13
    Finished epoch 14
    Finished epoch 15
    Finished epoch 16
    Finished epoch 17
    Finished epoch 18
    Finished epoch 19
    Finished epoch 20

Evaluating the trained model
----------------------------

Finally we will evaluate the model using :math:`ndcg@10` on the test set. To do
so we iterate over the test set in batches and compute :math:`ndcg@10` on each
batch. To compute the average :math:`ndcg@10` on the test set we take the sum
of all scores and finally divide by the length of the test set.

.. code-block:: python

    >>> from pytorchltr.evaluation import ndcg
    >>> loader = torch.utils.data.DataLoader(
    >>>   test, batch_size=16, collate_fn=test.collate_fn())
    >>> final_score = 0.0
    >>> for batch in loader:
    >>>   xs, ys, n = batch.features, batch.relevance, batch.n
    >>>   ndcg_score = ndcg(model(xs), ys, n, k=10)
    >>>   final_score += float(torch.sum(ndcg_score))

    >>> print("ndcg@10 on test set: %f" % (final_score / len(test)))
    ndcg@10 on test set: 0.445652

Additional information about available evaluation metrics and how to integrate
with :code:`pytrec_eval` can be found :ref:`here <evaluation>`.
