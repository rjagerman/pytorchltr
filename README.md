# PyTorch Learning to Rank (LTR)

[![Build](https://img.shields.io/travis/rjagerman/pytorchltr/master)](https://travis-ci.org/rjagerman/pytorchltr)
[![Documentation](https://img.shields.io/readthedocs/pytorchltr)](https://pytorchltr.readthedocs.io/)
[![Coverage](https://img.shields.io/codecov/c/github/rjagerman/pytorchltr/master)](https://codecov.io/gh/rjagerman/pytorchltr)
[![CodeFactor](https://img.shields.io/codefactor/grade/github/rjagerman/pytorchltr/master)](https://www.codefactor.io/repository/github/rjagerman/pytorchltr)
[![License](https://img.shields.io/github/license/rjagerman/pytorchltr)](https://github.com/rjagerman/pytorchltr/blob/master/LICENSE.md)

This is a library for Learning to Rank (LTR) with PyTorch.
The goal of this library is to support the infrastructure necessary for performing LTR experiments in PyTorch.

## Installation

In your virtualenv simply run:

    pip install pytorchltr 

Note that this library requires Python 3.6 or higher.

## Documentation

Documentation is available [here](https://pytorchltr.readthedocs.io/).

## Example

See `examples/01-basic-usage.py` for a more complete example including evaluation

```python
import torch
from pytorchltr.datasets import Example3
from pytorchltr.loss import PairwiseHingeLoss

# Load dataset
train = Example3(split="train")
collate_fn = train.collate_fn()

# Setup model, optimizer and loss
model = torch.nn.Linear(train[0]["features"].shape[1], 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss = PairwiseHingeLoss()

# Train for 3 epochs
for epoch in range(3):
    loader = torch.utils.data.DataLoader(train, batch_size=2, collate_fn=collate_fn)
    for batch in loader:
        xs, ys, n = batch["features"], batch["relevance"], batch["n"]
        l = loss(model(xs), ys, n).mean()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
```

## Dataset Disclaimer
This library provides utilities to automatically download and prepare several public LTR datasets.
We cannot vouch for the quality, correctness or usefulness of these datasets.
We do not host or distribute these datasets and it is ultimately **your responsibility** to determine whether you have permission to use each dataset under its respective license.

## Citing
If you find this software useful for your research, we kindly ask you to cite the following publication:

    @inproceedings{jagerman2020accelerated,
        author = {Jagerman, Rolf and de Rijke, Maarten},
        title = {Accelerated Convergence for Counterfactual Learning to Rank},
        year = {2020},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
        doi = {10.1145/3397271.3401069},
        series = {SIGIR’20}
    }
