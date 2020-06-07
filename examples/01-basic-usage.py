#!/usr/bin/env python
#
# This script trains and evaluates a linear ranker on the Example3 toy dataset.
#
# Usage with expected output:
#
#     $ PYTHONPATH=. python examples/01-basic-usage.py
#     [INFO, file] checking dataset files in './datasets/example3'
#     [WARNING, file] dataset file(s) in './datasets/example3' are missing or corrupt
#     [INFO, downloader] starting download from 'http://download.joachims.org/svm_light/examples/example3.tar.gz' to './datasets/example3/example3.tar.gz'
#     finished downloading [307B]
#     [INFO, file] extracting tar file at './datasets/example3/example3.tar.gz' to './datasets/example3'
#     [INFO, file] successfully checked all dataset files
#     [INFO, svmrank] loading svmrank dataset from ./datasets/example3/example3/train.dat
#     [INFO, file] checking dataset files in './datasets/example3'
#     [INFO, file] successfully checked all dataset files
#     [INFO, svmrank] loading svmrank dataset from ./datasets/example3/example3/test.dat
#     [INFO, 01-basic-usage] Test nDCG at start: 0.8617
#     [INFO, 01-basic-usage] Test nDCG after epoch 1: 0.8617
#     [INFO, 01-basic-usage] Test nDCG after epoch 2: 1.0000
#     [INFO, 01-basic-usage] Test nDCG after epoch 3: 1.0000

import torch
from pytorchltr.datasets import Example3
from pytorchltr.evaluation.dcg import ndcg
from pytorchltr.loss import PairwiseHingeLoss
import logging


# Setup logging
logging.basicConfig(
    format="[%(levelname)s, %(module)s] %(message)s",
    level=logging.INFO)

# Seed randomness
torch.manual_seed(42)

# Load the example3 toy dataset
train = Example3(split="train")
test = Example3(split="test")
collate_fn = train.collate_fn()

# Create model, optimizer and loss to optimize
model = torch.nn.Linear(train[0]["features"].shape[1], 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = PairwiseHingeLoss()


# Function to evaluate the model on the test split of the dataset
def evaluate():
    model.eval()
    loader = torch.utils.data.DataLoader(test, batch_size=2, shuffle=True,
                                         collate_fn=test.collate_fn())
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
                                         collate_fn=train.collate_fn())
    for batch in loader:
        xs, ys, n = batch["features"], batch["relevance"], batch["n"]
        loss = loss_fn(model(xs), ys, n).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logging.info("Test nDCG after epoch %d: %.4f" % (epoch + 1, evaluate()))
