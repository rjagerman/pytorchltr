import torch as _torch
from pytorchltr.utils import batch_pairs as _batch_pairs
from pytorchltr.utils import rank_by_score as _rank_by_score


class PairwiseLambdaLoss(_torch.nn.Module):
    """Pairwise LambdaLoss.

    Implementation of pairwise LambdaLoss.
    """
    def __init__(self, sigma=1.0):
        """
        Args:
            sigma: Steepness of the logistic curve.
        """
        super().__init__()
        self.sigma = sigma

    def loss_per_doc_pair(self, score_pairs, rel_pairs, n):
        """Computes a loss on given score pairs and relevance pairs.

        Args:
            score_pairs: A tensor of shape (batch_size, list_size,
                list_size, 2), where each entry (:, i, j, :) indicates a pair
                of scores for doc i and j.
            rel_pairs: A tensor of shape (batch_size, list_size, list_size, 2),
                where each entry (:, i, j, :) indicates the relevance
                for doc i and j.
            n: A batch of per-query number of documents (for padding purposes).

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

        # Compute pairwise differences for scores and relevances.
        score_pairs = _batch_pairs(scores)
        rel_pairs = _batch_pairs(relevance)

        # Compute loss per doc pair.
        loss_pairs = self.loss_per_doc_pair(score_pairs, rel_pairs, n)

        # Mask out padded documents per query in the batch
        n_grid = n[:, None, None].repeat(1, score_pairs.shape[1],
                                         score_pairs.shape[2])
        arange = _torch.arange(score_pairs.shape[1],
                               device=score_pairs.device)
        range_grid = _torch.max(*_torch.meshgrid([arange, arange]))
        range_grid = range_grid[None, :, :].repeat(n.shape[0], 1, 1)
        loss_pairs[n_grid <= range_grid] = 0.0
        print(loss_pairs)

        # Reduce final list loss from per doc pair loss to a per query loss.
        loss = self.loss_reduction(loss_pairs)

        # Return loss
        return loss


class PairwiseLambdaARPLoss1(PairwiseLambdaLoss):
    r"""ARP Loss 1:

    $$
    l(\mathbf{s}, \mathbf{y}) = -\sum_{i=1}^n \sum_{j=1}^n \log_2 \sum_{\pi}
    \left(\frac{1}{1 + e^{-\sigma (s_{\pi_i} - s_{\pi_j})}}\right)^{y_{\pi_i}}
    H(\pi \mid \mathbf{s})
    $$

    """
    def loss_per_doc_pair(self, score_pairs, rel_pairs, n):
        score_diffs = score_pairs[:, :, :, 0] - score_pairs[:, :, :, 1]
        sigmoid = (1.0 / (1.0 + _torch.exp(-self.sigma * score_diffs)))
        return -(_torch.log2(sigmoid ** rel_pairs[:, :, :, 0]))


class PairwiseLambdaARPLoss2(PairwiseLambdaLoss):
    r"""
    ARP Loss 2:

    $$
    l(\mathbf{s}, \mathbf{y}) = \sum_{y_i > y_j} |y_i - y_j| \log_2 \left(
    1 + e^{-\sigma(s_i - s_j)}
    \right)
    $$
    """
    def loss_per_doc_pair(self, score_pairs, rel_pairs, n):
        score_diffs = score_pairs[:, :, :, 0] - score_pairs[:, :, :, 1]
        rel_diffs = rel_pairs[:, :, :, 0] - rel_pairs[:, :, :, 1]
        loss = _torch.log2(1.0 + _torch.exp(-self.sigma * score_diffs))
        loss[rel_diffs <= 0] = 0.0
        return rel_diffs * loss


class PairwiseLambdaNDCGLoss1(PairwiseLambdaLoss):
    r"""
    NDCG Loss 1:

    $$
    l(\mathbf{s}, \mathbf{y}) = -\sum_{i=1}^n \sum_{j=1}^n \log_2 \sum_{\pi}
    \left(
        \frac{1}{1 + e^{-\sigma (s_{\pi_i} - s_{\pi_j})}}
    \right)^{\frac{G_{\pi_i}}{D_{\pi_i}}}
    H(\pi \mid \mathbf{s})
    $$
    where \(G_i = \frac{2^{y_i} - 1}{\text{maxDCG}}\) and
    \(D_i = \log_2(1 + i)\).
    """
    def loss_per_doc_pair(self, score_pairs, rel_pairs, n):
        score_diffs = score_pairs[:, :, :, 0] - score_pairs[:, :, :, 1]
        gains = _ndcg_gains(score_pairs, rel_pairs, n)[:, :, :, 0]
        arange = _torch.arange(score_pairs.shape[1],
                               device=score_pairs.device)
        discounts = _torch.log2(2.0 + arange)
        exponent = gains / discounts[None, :, None]
        sigmoid = (1.0 / (1.0 + _torch.exp(-self.sigma * score_diffs)))
        return -(_torch.log2(sigmoid ** exponent))


class PairwiseLambdaNDCGLoss2(PairwiseLambdaLoss):
    r"""
    NDCG Loss 2:

    $$
    l(\mathbf{s}, \mathbf{y}) = \sum_{y_i > y_j} \log_2 \sum_{\pi}
    \left(
    \frac{1}{1 + e^{-\sigma (s_i - s_j)}}
    \right)^{\delta_{ij} | G_i - G_j |}
    H(\pi \mid \mathbf{s})
    $$
    """
    def loss_per_doc_pair(self, score_pairs, rel_pairs, n):
        # Compute diffs for different parts of the loss function
        score_diffs = score_pairs[:, :, :, 0] - score_pairs[:, :, :, 1]
        rel_diffs = rel_pairs[:, :, :, 0] - rel_pairs[:, :, :, 1]
        gains = _ndcg_gains(score_pairs, rel_pairs, n)
        gain_diffs = gains[:, :, :, 0] - gains[:, :, :, 1]

        # Compute delta_{i, j} tensor
        arange = _torch.arange(score_pairs.shape[1] + 1,
                               device=score_pairs.device)
        discounts = _torch.log2(2.0 + arange)
        idx1 = _torch.abs(arange[:-1, None] - arange[None, :-1])
        idx2 = idx1 + 1
        delta = _torch.abs(1.0 / discounts[idx1] - 1.0 / discounts[idx2])

        # Compute final loss
        exponent = delta[None, :, :] * _torch.abs(gain_diffs)
        sigmoid = (1.0 / (1.0 + _torch.exp(-self.sigma * score_diffs)))
        loss = _torch.log2(sigmoid ** exponent)
        loss[rel_diffs <= 0] = 0.0
        return -loss


def _ndcg_gains(score_pairs, rel_pairs, n, exp=True):
    gains = rel_pairs[:, :, :, :]
    if exp:
        gains = (2 ** gains) - 1.0
    max_dcg = _max_dcg(rel_pairs[:, :, 0, 0], n, exp)
    max_dcg[max_dcg == 0.0] = 1.0
    return gains / max_dcg[:, None]


def _max_dcg(relevance, n, exp=True):
    ranking = _rank_by_score(relevance.double(), n)
    arange = _torch.arange(ranking.shape[1],
                           device=relevance.device)
    discounts = _torch.log2(2.0 + arange)
    gains = _torch.gather(relevance, 1, ranking)
    gains[n[:, None] <= arange[None, :]] = 0.0
    if exp:
        gains = (2 ** gains) - 1.0
    return _torch.sum(gains / discounts[None, :], dim=1)
