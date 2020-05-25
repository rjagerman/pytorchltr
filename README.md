# PyTorch Learning to Rank (LTR)

[![Build Status](https://travis-ci.org/rjagerman/pytorchltr.svg?branch=master)](https://travis-ci.org/rjagerman/pytorchltr)
[![codecov](https://codecov.io/gh/rjagerman/pytorchltr/branch/master/graph/badge.svg)](https://codecov.io/gh/rjagerman/pytorchltr)

> :warning: **This is an early release and subject to breaking changes**

This is a library for Learning to Rank (LTR) with PyTorch.
The goal of this library is to support the infrastructure necessary for performing LTR experiments in PyTorch.

## Installation

In your virtualenv simply run:

    pip install git+ssh://git@github.com/rjagerman/pytorchltr.git#egg=pytorchltr

Note that this library requires Python 3.6 or higher.

## Example

See `examples/01-basic-usage.py` for a more complete example including evaluation

    import torch
    from pytorchltr.dataset.resources import Example3
    from pytorchltr.loss.pairwise import AdditivePairwiseLoss
    from pytorchltr.dataset.svmrank import create_svmranking_collate_fn

    # Load dataset
    dataset = Example3("./datasets/example3", download=True)
    train, collate_fn = dataset.train(), dataset.collate_fn()

    # Setup model, optimizer and loss
    model = torch.nn.Linear(train[0]["features"].shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = AdditivePairwiseLoss()

    # Train for 3 epochs
    for epoch in range(3):
        loader = torch.utils.data.DataLoader(train, batch_size=2, collate_fn=collate_fn)
        for batch in loader:
            xs, ys, n = batch["features"], batch["relevance"], batch["n"]
            l = loss(model(xs), ys, n).mean()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

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
