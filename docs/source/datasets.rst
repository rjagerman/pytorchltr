.. _datasets:

Datasets
========

PyTorchLTR provides several LTR datasets utility classes that can be used to
automatically process and/or download the dataset files.

.. warning::

    PyTorchLTR provides utilities to automatically download and prepare several
    public LTR datasets. We cannot vouch for the quality, correctness or
    usefulness of these datasets. We do not host or distribute any datasets and
    it is ultimately **your responsibility** to determine whether you have
    permission to use each dataset under its respective license.

Example
-------

The following is a usage example for the small Example3 dataset.

.. code-block:: python

    >>> from pytorchltr.datasets import Example3
    >>> train = Example3(split="train")
    >>> test = Example3(split="test")
    >>> print(len(train))
    3
    >>> print(len(test))
    1
    >>> sample = train[0]
    >>> print(sample["features"])
    tensor([[1.0000, 1.0000, 0.0000, 0.3333, 0.0000],
            [0.0000, 0.0000, 1.0000, 0.0000, 1.0000],
            [0.0000, 1.0000, 0.0000, 1.0000, 0.0000],
            [0.0000, 0.0000, 1.0000, 0.6667, 0.0000]])
    >>> print(sample["relevance"])
    tensor([3, 2, 1, 1])
    >>> print(sample["n"])
    4

.. note::

    PyTorchLTR looks for dataset files in (and downloads them to) the following
    locations:

    * The :code:`location` arg if it is specified in the constructor of each
      respective Dataset class.
    * :code:`$PYTORCHLTR_DATASET_PATH/{dataset_name}` if
      :code:`$PYTORCHLTR_DATASET_PATH` is a defined environment variable.
    * :code:`$DATASET_PATH/{dataset_name}` if :code:`$DATASET_PATH` is a defined
      environment variable.
    * :code:`$HOME/.pytorchltr_datasets/{dataset_name}` if all the above fail.


SVMRank datasets
----------------

Example3
^^^^^^^^
.. autoclass:: pytorchltr.datasets.Example3
   :members:

   .. automethod:: __init__
   .. automethod:: collate_fn
   .. automethod:: __getitem__
   .. automethod:: __len__

Istella-S
^^^^^^^^^
.. autoclass:: pytorchltr.datasets.IstellaS
   :members:

   .. automethod:: __init__
   .. automethod:: collate_fn
   .. automethod:: __getitem__
   .. automethod:: __len__

MSLR-WEB10K
^^^^^^^^^^^
.. autoclass:: pytorchltr.datasets.MSLR10K
   :members:

   .. automethod:: __init__
   .. automethod:: collate_fn
   .. automethod:: __getitem__
   .. automethod:: __len__

MSLR-WEB30K
^^^^^^^^^^^
.. autoclass:: pytorchltr.datasets.MSLR30K
   :members:

   .. automethod:: __init__
   .. automethod:: collate_fn
   .. automethod:: __getitem__
   .. automethod:: __len__
