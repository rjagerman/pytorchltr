import _torch


class PairwiseLoss(_torch.nn.Module):
    def __init__(self):
        super().__init__(self)

    def _batch_pairwise_difference(self, x):
        """Returns a pairwise difference matrix p.

        This matrix contains entries p_{i,j} = x_i - x_j

        Arguments:
            x: The input batch of dimension (batch_size, list_size, 1).
        """

        # Construct broadcasted x_{:,i,0...list_size}
        x_ij = torch.repeat_interleave(x, x.shape[1], dim=2)

        # Construct broadcasted x_{:,0...list_size,i}
        x_ji = torch.repeat_interleave(x.permute(0, 2, 1), x.shape[1], dim=1)

        # Compute difference
        return x_ij - x_ji
