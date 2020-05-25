#!/usr/bin/env python
#
# This script trains and evaluates a linear ranker on the Example3 toy dataset.
#
# Usage with expected output:
#
#     $ PYTHONPATH=. python examples/01-basic-usage.py
#     [INFO, resource] downloading example3 dataset from http://download.joachims.org/svm_light/examples/example3.tar.gz to ./datasets/example3/data.tar.gz
#     [INFO, resource] extracting example3 dataset from ./datasets/example3/data.tar.gz to ./datasets/example3
#     [INFO, resource] loading svmrank dataset from ./datasets/example3/example3/train.dat
#     [INFO, resource] loading svmrank dataset from ./datasets/example3/example3/test.dat
#     [INFO, 01-basic-usage] Test nDCG at start: 0.8617
#     [INFO, 01-basic-usage] Test nDCG after epoch 1: 0.8617
#     [INFO, 01-basic-usage] Test nDCG after epoch 2: 1.0000
#     [INFO, 01-basic-usage] Test nDCG after epoch 3: 1.0000

import torch
from pytorchltr.dataset.resources import Example3
from pytorchltr.evaluation.dcg import ndcg
from pytorchltr.loss.pairwise import AdditivePairwiseLoss
import logging


# Setup logging
logging.basicConfig(
    format="[%(levelname)s, %(module)s] %(message)s",
    level=logging.INFO)

# Seed randomness
torch.manual_seed(42)

# Load the example3 toy dataset
dataset = Example3("./datasets/example3", download=True)
train, test, collate_fn = dataset.train(), dataset.test(), dataset.collate_fn()

# Create model, optimizer and loss to optimize
model = torch.nn.Linear(train[0]["features"].shape[1], 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss = AdditivePairwiseLoss()

# Function to evaluate the model on the test split of the dataset
def evaluate():
    model.eval()
    loader = torch.utils.data.DataLoader(test, batch_size=2, shuffle=True,
                                         collate_fn=collate_fn)
    ndcg_score = 0.0
    for batch in loader:
        xs, ys, n = batch["features"], batch["relevance"], batch["n"]
        ndcg_score += float(torch.sum(ndcg(model(xs), ys, n, k=10)))

    ndcg_score /= len(test)
    model.train()
    return ndcg_score

logging.info("Test nDCG at start: %.4f" % evaluate())

# Train model for 3 epochs
for epoch in range(3):
    loader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True,
                                         collate_fn=collate_fn)
    for batch in loader:
        xs, ys, n = batch["features"], batch["relevance"], batch["n"]
        l = loss(model(xs), ys, n).mean()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    logging.info("Test nDCG after epoch %d: %.4f" % (epoch + 1, evaluate()))
