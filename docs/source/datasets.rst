Datasets
========

PyTorchLTR provides several LTR datasets utility classes that can be used to
automatically process and/or download the dataset files.

PyTorchLTR looks for dataset files in (and downloads them to) the following
locations:

  * The :code:`location` arg if it is specified in the constructor of the
    Dataset class.
  * :code:`$PYTORCHLTR_DATASET_PATH/{dataset_name}` if
    :code:`$PYTORCHLTR_DATASET_PATH` is a defined environment variable.
  * :code:`$DATASET_PATH/{dataset_name}` if :code:`$DATASET_PATH` is a defined
    environment variable.
  * :code:`$HOME/.pytorchltr_datasets/{dataset_name}` if all the above fail.

SVMRank datasets
----------------
.. autoclass:: pytorchltr.datasets.Example3
   :members:

   .. automethod:: __init__
   .. automethod:: collate_fn
   .. automethod:: __getitem__
   .. automethod:: __len__

.. autoclass:: pytorchltr.datasets.IstellaS
   :members:

   .. automethod:: __init__
   .. automethod:: collate_fn
   .. automethod:: __getitem__
   .. automethod:: __len__

.. autoclass:: pytorchltr.datasets.MSLR10K
   :members:

   .. automethod:: __init__
   .. automethod:: collate_fn
   .. automethod:: __getitem__
   .. automethod:: __len__

.. autoclass:: pytorchltr.datasets.MSLR30K
   :members:

   .. automethod:: __init__
   .. automethod:: collate_fn
   .. automethod:: __getitem__
   .. automethod:: __len__
