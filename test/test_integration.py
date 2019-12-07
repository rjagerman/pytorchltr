from io import BytesIO

from test.dataset.test_svmrank import write_dataset_file
from pytorchltr.dataset.svmrank import create_svmranking_collate_fn
from pytorchltr.dataset.svmrank import svmranking_dataset
from pytorchltr.loss.pairwise import AdditivePairwiseLoss
from pytorchltr.evaluation.arp import arp
from pytest import approx

import torch


class Model(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.ff = torch.nn.Linear(in_features, 1)

    def forward(self, xs):
        return self.ff(xs)


def test_basic_sgd_learning():
    torch.manual_seed(42)
    with BytesIO() as file:
       write_dataset_file(file)
       dataset = svmranking_dataset(file)

    with BytesIO() as file:
       write_dataset_file(file)
       vali_dataset = svmranking_dataset(file)

    input_dim = dataset[0]["features"].shape[1]
    collate_fn = create_svmranking_collate_fn(max_list_size=50)
    model = Model(input_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss = AdditivePairwiseLoss()
    arp_per_epoch = torch.zeros(100)

    # Perform 100 epochs
    for epoch in range(100):

        # Load and iterate over dataset
        avg_arp = 0.0
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        for i, batch in enumerate(loader):
            # Get batch of samples
            xs, ys, n = batch["features"], batch["relevance"], batch["n"]

            # Compute loss
            l = loss(model(xs), ys, n)
            l = l.mean()

            # Perform SGD step
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            # Evaluate ARP on train data
            model.eval()
            arp_score = torch.mean(arp(model(xs), ys, n))
            model.train()

            # Keep track of average ARP in this epoch
            avg_arp = avg_arp + (float(arp_score) - avg_arp) / (i + 1)

        # Record the average ARP.
        arp_per_epoch[epoch] = avg_arp

    # Assert that the ARP was decreased by a significant amount from start to
    # finish.
    assert arp_per_epoch[-1] - arp_per_epoch[0] <= -0.40
