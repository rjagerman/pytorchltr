import torch as _torch
from pytorchltr.utils import batch_pairs as _batch_pairs
from pytorchltr.utils import rank_by_score as _rank_by_score


class PairwiseLambdaLoss(_torch.nn.Module):
    """Pairwise LambdaLoss.

    Implementation of pairwise LambdaLoss.
    """
    def __init__(self, sigma=1.0):
        """Initializes the Pairwise LambdaLoss."""
        super().__init__()
        self.sigma = sigma

    def forward(self, scores, relevance, n):
        """Computes the loss for given batch of samples.

        Args:
            scores: A batch of per-query-document scores.
            relevance: A batch of per-query-document relevance labels.
            n: A batch of per-query number of documents (for padding purposes).
        """
        # Reshape relevance if necessary.
        if relevance.ndimension() == 2:
            relevance = relevance.reshape(
                (relevance.shape[0], relevance.shape[1], 1))
        if scores.ndimension() == 2:
            scores = scores.reshape((scores.shape[0], scores.shape[1], 1))

        # Compute ranking and sort scores and relevance
        ranking = _rank_by_score(scores, n)
        ranking = ranking.view((ranking.shape[0], ranking.shape[1], 1))
        scores = _torch.gather(scores, 1, ranking)
        relevance = _torch.gather(relevance, 1, ranking)

        # To do: compute arange for NDCG-Loss
        # To do: compute batch_pairs for both ARP-Loss and NDCG-Loss

        # Return loss
        return None
