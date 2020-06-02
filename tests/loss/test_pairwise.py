import torch
from pytorchltr.loss.pairwise import PairwiseHingeLoss
from pytest import approx


def test_pairwise_rank_perfect():
    loss_fn = PairwiseHingeLoss()
    scores = torch.FloatTensor([[[0.0], [0.0], [1.0], [2.0], [1.0]]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # RankLoss: 1.0 + \sum_{(d_i, d_j) : y_i > y_j} hinge(f(d_j) - f(d_i)))
    assert loss.item() == approx(0.0)


def test_pairwise_rank_2():
    loss_fn = PairwiseHingeLoss()
    scores = torch.FloatTensor([[[0.0], [0.0], [1.0], [1.0], [1.0]]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # RankLoss: 1.0 + \sum_{(d_i, d_j) : y_i > y_j} hinge(f(d_j) - f(d_i)))
    assert loss.item() == approx(2.0)


def test_pairwise_rank_3():
    loss_fn = PairwiseHingeLoss()
    scores = torch.FloatTensor([[[0.0], [0.0], [1.0], [-5.0], [1.0]]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # RankLoss: 1.0 + \sum_{(d_i, d_j) : y_i > y_j} hinge(f(d_j) - f(d_i)))
    assert loss.item() == approx(7.0 + 7.0 + 6.0 + 6.0)


def test_pairwise_rank_batch():
    loss_fn = PairwiseHingeLoss()
    scores = torch.FloatTensor([
        [[0.0], [10.0], [1.0], [0.5], [1.0]],
        [[1.0], [3.5], [6.0], [4.3], [10.0]]])
    ys = torch.LongTensor([
        [0, 2, 1, 2, 1],
        [1, 2, 2, 1, 0]])
    n = torch.LongTensor([5, 4])
    loss = loss_fn(scores, ys, n)

    # RankLoss: 1.0 + \sum_{(d_i, d_j) : y_i > y_j} hinge(f(d_j) - f(d_i)))
    assert loss[0].item() == approx(0.5 + 1.5 + 1.5)
    assert loss[1].item() == approx(5.3 - 3.5)


def test_pairwise_rank_cutoff():
    loss_fn = PairwiseHingeLoss()
    scores = torch.FloatTensor([[[1.0], [3.5], [6.0], [4.3], [8.0]]])
    ys = torch.LongTensor([[1, 2, 2, 1, 0]])
    n1 = torch.LongTensor([3])
    n2 = torch.LongTensor([4])
    n3 = torch.LongTensor([5])
    loss1 = loss_fn(scores, ys, n1)
    loss2 = loss_fn(scores, ys, n2)
    loss3 = loss_fn(scores, ys, n3)

    # RankLoss: 1.0 + \sum_{(d_i, d_j) : y_i > y_j} hinge(f(d_j) - f(d_i)))
    assert loss1.item() == approx(0.0)
    assert loss2.item() == approx(5.3 - 3.5)
    assert loss3.item() == approx(
        (5.3 - 3.5) + (9.0 - 4.3) + (9.0 - 6.0) + (9.0 - 3.5) + (9.0 - 1.0))
