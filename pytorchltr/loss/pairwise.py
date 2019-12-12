"""Pairwise ranking losses."""
import torch as _torch


class AdditivePairwiseLoss(_torch.nn.Module):
    """Additive pairwise ranking losses.

    Implementation of linearly decomposible additive pairwise ranking losses.
    This includes RankSVM hinge loss and variations.
    """
    def __init__(self, loss_modifier="rank"):
        """Initializes the Additive Pairwise Loss.

        Arguments:
            loss_modifier: One of "rank", "normrank" or "dcg", where "rank"
                indicates the hinge loss upper bound to svmrank, "normrank"
                normalizes the "rank" objective with size of the ranked list
                and "dcg" applies additive dcg weighting.
        """
        super().__init__()
        self.loss_modifier = loss_modifier

    def forward(self, scores, relevance, n):
        """Computes the loss for given batch of samples.

        Arguments:
            scores: A batch of per-query-document scores.
            relevance: A batch of per-query-document relevance labels.
            n: A batch of per-query number of documents (for padding purposes).
        """
        # Reshape relevance if necessary
        if relevance.ndimension() == 2:
            relevance = relevance.reshape((relevance.shape[0], relevance.shape[1], 1))

        # Compute pairwise differences for scores
        s_ij = _batch_pairwise_difference(scores)

        # Compute hinge loss for pairs that have a relevance difference
        loss = 1.0 - s_ij
        loss[_batch_pairwise_difference(relevance) <= 0.0] = 0.0
        loss[loss < 0.0] = 0.0

        # Mask out padded documents per query in the batch
        n_grid = n[:, None, None].repeat(1, s_ij.shape[1], s_ij.shape[2])
        arange = _torch.arange(s_ij.shape[1], device=s_ij.device)
        range_grid_1 = _torch.max(*_torch.meshgrid([arange, arange]))
        range_grid = range_grid_1[None, :, :].repeat(n.shape[0], 1, 1)
        loss[n_grid <= range_grid] = 0.0

        # Reduce final list loss by sum, creating upper bound on rel result ranks
        loss = loss.view(loss.shape[0], -1)
        loss = loss.sum(1)

        # Apply a loss modifier
        if self.loss_modifier == "normrank":
            loss /= n.to(dtype=loss.dtype)
        elif self.loss_modifier == "dcg":
            loss = -1.0 / _torch.log(2.0 + loss)

        # Return loss
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
