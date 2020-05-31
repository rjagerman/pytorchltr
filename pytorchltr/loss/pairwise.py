"""Pairwise ranking losses."""
import torch as _torch


class AdditivePairwiseLoss(_torch.nn.Module):
    """Additive pairwise ranking losses.

    Implementation of linearly decomposible additive pairwise ranking losses.
    This includes RankSVM hinge loss and variations.
    """
    def __init__(self):
        """Initializes the Additive Pairwise Loss."""
        super().__init__()

    def loss_per_doc_pair(self, score_pairs, rel_pairs):
        """Computes a loss on given score pairs and relevance pairs.

        Args:
            score_pairs: A tensor of shape (batch_size, list_size, list_size)
                where each entry indicates the score difference of doc i and j.
            rel_pairs: A tensor of shape (batch_size, list_size, list_size)
                where each entry indicates the relevance difference of doc i
                and j.

        Returns:
            A tensor of shape (batch_size, list_size, list_size) with a loss
            per document pair.
        """
        raise NotImplementedError

    def loss_reduction(self, loss):
        """Reduces the paired loss to a per sample loss.

        Args:
            loss: A tensor of shape (batch_size, list_size, list_size)
                where each entry indicates the loss of doc pair i and j.

        Returns:
            A tensor of shape (batch_size) indicating the loss per training
            sample.
        """
        return loss.view(loss.shape[0], -1).sum(1)

    def loss_modifier(self, loss):
        """A modifier to apply to the loss."""
        return loss

    def forward(self, scores, relevance, n):
        """Computes the loss for given batch of samples.

        Arguments:
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
        score_pairs = _batch_pairwise_difference(scores)
        rel_pairs = _batch_pairwise_difference(relevance)

        # Compute loss per doc pair.
        loss = self.loss_per_doc_pair(score_pairs, rel_pairs)

        # Mask out padded documents per query in the batch
        n_grid = n[:, None, None].repeat(1, score_pairs.shape[1],
                                         score_pairs.shape[2])
        arange = _torch.arange(score_pairs.shape[1], device=score_pairs.device)
        range_grid = _torch.max(*_torch.meshgrid([arange, arange]))
        range_grid = range_grid[None, :, :].repeat(n.shape[0], 1, 1)
        loss[n_grid <= range_grid] = 0.0

        # Reduce final list loss from per doc pair loss to a per query loss.
        loss = self.loss_reduction(loss)

        # Apply a loss modifier.
        loss = self.loss_modifier(loss)

        # Return loss
        return loss


class PairwiseHingeLoss(AdditivePairwiseLoss):
    """Pairwise hinge loss formulation of SVMRank."""
    def loss_per_doc_pair(self, score_pairs, rel_pairs):
        loss = 1.0 - score_pairs
        loss[rel_pairs <= 0.0] = 0.0
        loss[loss < 0.0] = 0.0
        return loss


class PairwiseDCGHingeLoss(PairwiseHingeLoss):
    """DCG-modified pairwise hinge loss."""
    def loss_modifier(self, loss):
        return -1.0 / _torch.log(2.0 + loss)


class PairwiseLogisticLoss(AdditivePairwiseLoss):
    """Pairwise logistic loss formulation of RankNet."""
    def __init__(self, sigma=1.0):
        """
        Args:
            sigma: A hyper parameter indicating the sharpness of the logistic
                curve used.
        """
        self.sigma = sigma

    def loss_per_doc_pair(self, score_pairs, rel_pairs):
        loss = _torch.log2(1.0 + _torch.exp(-self.sigma * score_pairs))
        loss[rel_pairs <= 0.0] = 0.0
        return loss


def _batch_pairwise_difference(x):
    """Returns a pairwise difference matrix p.

    This matrix contains entries p_{i,j} = x_i - x_j

    Arguments:
        x: The input batch of dimension (batch_size, list_size, 1).

    Returns:
        A tensor of size (batch_size, list_size, list_size) containing pairwise
        differences.
    """

    # Construct broadcasted x_{:,i,0...list_size}
    x_ij = _torch.repeat_interleave(x, x.shape[1], dim=2)

    # Construct broadcasted x_{:,0...list_size,i}
    x_ji = _torch.repeat_interleave(x.permute(0, 2, 1), x.shape[1], dim=1)

    # Compute difference
    return x_ij - x_ji
