import torch
from pytorchltr.loss.pairwise_additive import PairwiseHingeLoss
from pytorchltr.loss.pairwise_additive import PairwiseDCGHingeLoss
from pytorchltr.loss.pairwise_additive import PairwiseLogisticLoss
from math import log
from math import log2
from math import exp
from pytest import approx


def test_pairwise_hinge_autoreshape_scores():
    loss_fn = PairwiseHingeLoss()
    scores = torch.FloatTensor([[0.0, 0.0, 1.0, 2.0, 1.0]])
    ys = torch.LongTensor([[[0], [0], [1], [2], [1]]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # Hinge loss: 1.0 + \sum_{(d_i, d_j) : y_i > y_j} hinge(f(d_j) - f(d_i)))
    assert loss.item() == approx(0.0)


def test_pairwise_hinge_autoreshape_relevance():
    loss_fn = PairwiseHingeLoss()
    scores = torch.FloatTensor([[[0.0], [0.0], [1.0], [2.0], [1.0]]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # Hinge loss: 1.0 + \sum_{(d_i, d_j) : y_i > y_j} hinge(f(d_j) - f(d_i)))
    assert loss.item() == approx(0.0)


def test_pairwise_hinge_perfect():
    loss_fn = PairwiseHingeLoss()
    scores = torch.FloatTensor([[0.0, 0.0, 1.0, 2.0, 1.0]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # Hinge loss: 1.0 + \sum_{(d_i, d_j) : y_i > y_j} hinge(f(d_j) - f(d_i)))
    assert loss.item() == approx(0.0)


def test_pairwise_hinge_2():
    loss_fn = PairwiseHingeLoss()
    scores = torch.FloatTensor([[[0.0], [0.0], [1.0], [1.0], [1.0]]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # Hinge loss: 1.0 + \sum_{(d_i, d_j) : y_i > y_j} hinge(f(d_j) - f(d_i)))
    assert loss.item() == approx(2.0)


def test_pairwise_hinge_3():
    loss_fn = PairwiseHingeLoss()
    scores = torch.FloatTensor([[[0.0], [0.0], [1.0], [-5.0], [1.0]]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # Hinge loss: 1.0 + \sum_{(d_i, d_j) : y_i > y_j} hinge(f(d_j) - f(d_i)))
    assert loss.item() == approx(7.0 + 7.0 + 6.0 + 6.0)


def test_pairwise_hinge_batch():
    loss_fn = PairwiseHingeLoss()
    scores = torch.FloatTensor([
        [[0.0], [10.0], [1.0], [0.5], [1.0]],
        [[1.0], [3.5], [6.0], [4.3], [10.0]]])
    ys = torch.LongTensor([
        [0, 2, 1, 2, 1],
        [1, 2, 2, 1, 0]])
    n = torch.LongTensor([5, 4])
    loss = loss_fn(scores, ys, n)

    # Hinge loss: 1.0 + \sum_{(d_i, d_j) : y_i > y_j} hinge(f(d_j) - f(d_i)))
    assert loss[0].item() == approx(0.5 + 1.5 + 1.5)
    assert loss[1].item() == approx(5.3 - 3.5)


def test_pairwise_hinge_cutoff():
    loss_fn = PairwiseHingeLoss()
    scores = torch.FloatTensor([[[1.0], [3.5], [6.0], [4.3], [8.0]]])
    ys = torch.LongTensor([[1, 2, 2, 1, 0]])
    n1 = torch.LongTensor([3])
    n2 = torch.LongTensor([4])
    n3 = torch.LongTensor([5])
    loss1 = loss_fn(scores, ys, n1)
    loss2 = loss_fn(scores, ys, n2)
    loss3 = loss_fn(scores, ys, n3)

    # Hinge loss: 1.0 + \sum_{(d_i, d_j) : y_i > y_j} hinge(f(d_j) - f(d_i)))
    assert loss1.item() == approx(0.0)
    assert loss2.item() == approx(5.3 - 3.5)
    assert loss3.item() == approx(
        (5.3 - 3.5) + (9.0 - 4.3) + (9.0 - 6.0) + (9.0 - 3.5) + (9.0 - 1.0))


def test_pairwise_dcghinge_perfect():
    loss_fn = PairwiseDCGHingeLoss()
    scores = torch.FloatTensor([[[0.0], [0.0], [1.0], [2.0], [1.0]]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # DCG-modified Hinge loss:
    # -1.0 / log2(1.0 + \sum_{(d_i, d_j) : y_i > y_j} hinge(f(d_j) - f(d_i))))
    assert loss.item() == approx(-1.0 / log(2.0 + 0.0))


def test_pairwise_dcghinge_worst():
    loss_fn = PairwiseDCGHingeLoss()
    scores = torch.FloatTensor([[[3.0], [3.0], [1.0], [0.0], [1.0]]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # DCG-modified Hinge loss:
    # -1.0 / log2(1.0 + \sum_{(d_i, d_j) : y_i > y_j} hinge(f(d_j) - f(d_i))))
    assert loss.item() == approx(-1.0 / log(2.0 + 24.0))


def test_pairwise_logistic_perfect():
    loss_fn = PairwiseLogisticLoss()
    scores = torch.FloatTensor([[[0.0], [0.0], [1.0], [2.0], [1.0]]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # DCG-modified Hinge loss:
    # \sum_{(d_i, d_j) : y_i > y_j} log2(1.0 + exp(-1.0 * (f(d_j) - f(d_i)))))
    d1 = (2.0 - 1.0)  # 2 times
    d2 = (2.0 - 0.0)  # 2 times
    d3 = (1.0 - 0.0)  # 4 times
    assert loss.item() == approx(
        log2(1.0 + exp(-1.0 * d1)) * 2 +
        log2(1.0 + exp(-1.0 * d2)) * 2 +
        log2(1.0 + exp(-1.0 * d3)) * 4)


def test_pairwise_logistic_worst():
    loss_fn = PairwiseLogisticLoss()
    scores = torch.FloatTensor([[[3.0], [3.0], [1.0], [0.0], [1.0]]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # Logistic loss:
    # \sum_{(d_i, d_j) : y_i > y_j} log2(1.0 + exp(-1.0 * (f(d_j) - f(d_i)))))
    d1 = (0.0 - 3.0)  # 2 times
    d2 = (0.0 - 1.0)  # 2 times
    d3 = (1.0 - 3.0)  # 4 times
    assert loss.item() == approx(
        log2(1.0 + exp(-1.0 * d1)) * 2 +
        log2(1.0 + exp(-1.0 * d2)) * 2 +
        log2(1.0 + exp(-1.0 * d3)) * 4)
