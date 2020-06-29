import torch as _torch
from pytorchltr.utils import batch_pairs


class PairwiseAdditiveLoss(_torch.nn.Module):
    """Pairwise additive ranking losses.

    Implementation of linearly decomposible additive pairwise ranking losses.
    This includes RankSVM hinge loss and variations.
    """
    def __init__(self):
        r""""""
        super().__init__()

    def loss_per_doc_pair(self, score_pairs, rel_pairs):
        """Computes a loss on given score pairs and relevance pairs.

        Args:
            score_pairs: A tensor of shape (batch_size, list_size,
                list_size, 2), where each entry (:, i, j, :) indicates a pair
                of scores for doc i and j.
            rel_pairs: A tensor of shape (batch_size, list_size, list_size, 2),
                where each entry (:, i, j, :) indicates the relevance
                for doc i and j.

        Returns:
            A tensor of shape (batch_size, list_size, list_size) with a loss
            per document pair.
        """
        raise NotImplementedError

    def loss_reduction(self, loss_pairs):
        """Reduces the paired loss to a per sample loss.

        Args:
            loss_pairs: A tensor of shape (batch_size, list_size, list_size)
                where each entry indicates the loss of doc pair i and j.

        Returns:
            A tensor of shape (batch_size) indicating the loss per training
            sample.
        """
        return loss_pairs.view(loss_pairs.shape[0], -1).sum(1)

    def loss_modifier(self, loss):
        """A modifier to apply to the loss."""
        return loss

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

        # Compute pairwise differences for scores and relevances.
        score_pairs = batch_pairs(scores)
        rel_pairs = batch_pairs(relevance)

        # Compute loss per doc pair.
        loss_pairs = self.loss_per_doc_pair(score_pairs, rel_pairs)

        # Mask out padded documents per query in the batch
        n_grid = n[:, None, None].repeat(1, score_pairs.shape[1],
                                         score_pairs.shape[2])
        arange = _torch.arange(score_pairs.shape[1],
                               device=score_pairs.device)
        range_grid = _torch.max(*_torch.meshgrid([arange, arange]))
        range_grid = range_grid[None, :, :].repeat(n.shape[0], 1, 1)
        loss_pairs[n_grid <= range_grid] = 0.0

        # Reduce final list loss from per doc pair loss to a per query loss.
        loss = self.loss_reduction(loss_pairs)

        # Apply a loss modifier.
        loss = self.loss_modifier(loss)

        # Return loss
        return loss


class PairwiseHingeLoss(PairwiseAdditiveLoss):
    r"""Pairwise hinge loss formulation of SVMRank:

    $$
    l(\mathbf{s}, \mathbf{y}) = \sum_{y_i > y _j} max\left(
        0, 1 - (s_i - s_j)
    \right)
    $$
    """
    def loss_per_doc_pair(self, score_pairs, rel_pairs):
        score_pair_diffs = score_pairs[:, :, :, 0] - score_pairs[:, :, :, 1]
        rel_pair_diffs = rel_pairs[:, :, :, 0] - rel_pairs[:, :, :, 1]
        loss = 1.0 - score_pair_diffs
        loss[rel_pair_diffs <= 0.0] = 0.0
        loss[loss < 0.0] = 0.0
        return loss


class PairwiseDCGHingeLoss(PairwiseHingeLoss):
    r"""Pairwise DCG-modified hinge loss:

    $$
    l(\mathbf{s}, \mathbf{y}) =
    \frac{-1}{\log\left(
        2 + \sum_{y_i > y_j}
        max\left(0, 1 - (s_i - s_j)\right)
    \right)}
    $$
    """
    def loss_modifier(self, loss):
        return -1.0 / _torch.log(2.0 + loss)


class PairwiseLogisticLoss(PairwiseAdditiveLoss):
    r"""Pairwise logistic loss formulation of RankNet:

    $$
    l(\mathbf{s}, \mathbf{y}) = \sum_{y_i > y_j} \log_2\left(1 + e^{
        -\sigma \left(s_i - s_j\right)
    }\right)
    $$
    """
    def __init__(self, sigma=1.0):
        """
        Args:
            sigma: Steepness of the logistic curve.
        """
        super().__init__()
        self.sigma = sigma

    def loss_per_doc_pair(self, score_pairs, rel_pairs):
        score_pair_diffs = score_pairs[:, :, :, 0] - score_pairs[:, :, :, 1]
        rel_pair_diffs = rel_pairs[:, :, :, 0] - rel_pairs[:, :, :, 1]
        loss = _torch.log2(1.0 + _torch.exp(-self.sigma * score_pair_diffs))
        loss[rel_pair_diffs <= 0.0] = 0.0
        return loss
