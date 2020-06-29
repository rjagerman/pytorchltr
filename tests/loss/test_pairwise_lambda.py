import torch
from pytorchltr.loss import LambdaARPLoss1
from pytorchltr.loss import LambdaARPLoss2
from pytorchltr.loss import LambdaNDCGLoss1
from pytorchltr.loss import LambdaNDCGLoss2
from math import log2
from math import exp
from pytest import approx


def test_lambda_losses_batch():
    scores = torch.tensor([
        [0.5, 2.0, 1.0],
        [0.9, -1.2, 0.0]
    ])
    relevance = torch.tensor([
        [2, 0, 1],
        [0, 1, 0]
    ])
    n = torch.tensor([3, 2])

    loss_fn = LambdaARPLoss1()
    loss = loss_fn(scores, relevance, n)
    assert float(loss[0]) == approx(13.298417091369629)
    assert float(loss[1]) == approx(4.196318626403809)

    loss_fn = LambdaARPLoss2()
    loss = loss_fn(scores, relevance, n)
    assert float(loss[0]) == approx(8.209173202514648)
    assert float(loss[1]) == approx(3.1963188648223877)

    loss_fn = LambdaNDCGLoss1()
    loss = loss_fn(scores, relevance, n)
    assert float(loss[0]) == approx(2.629549503326416)
    assert float(loss[1]) == approx(2.647582530975342)

    loss_fn = LambdaNDCGLoss2()
    loss = loss_fn(scores, relevance, n)
    assert float(loss[0]) == approx(0.3102627396583557)
    assert float(loss[1]) == approx(0.4184933304786682)


def test_lambda_arp1_reshape_scores():
    loss_fn = LambdaARPLoss1()
    scores = torch.FloatTensor([[0.0, 0.0, 1.0, 2.0, 1.0]])
    ys = torch.LongTensor([[[0], [0], [1], [2], [1]]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # Compute result iteratively
    expected = 0.0
    for i in range(n[0]):
        for j in range(n[0]):
            score_diff = float(scores[0, i] - scores[0, j])
            inner = 1.0 / (1.0 + exp(-1.0 * score_diff))
            inner = inner ** float(ys[0, i, 0])
            expected -= log2(inner)

    assert loss.item() == approx(expected)


def test_lambda_arp1_reshape_rel():
    loss_fn = LambdaARPLoss1()
    scores = torch.FloatTensor([[[0.0], [0.0], [1.0], [2.0], [1.0]]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # Compute result iteratively
    expected = 0.0
    for i in range(n[0]):
        for j in range(n[0]):
            inner = 1.0 / (1.0 + exp(
                -1.0 * float(scores[0, i, 0] - scores[0, j, 0])))
            inner = inner ** float(ys[0, i])
            expected -= log2(inner)

    assert loss.item() == approx(expected)


def test_lambda_arp1_perfect():
    loss_fn = LambdaARPLoss1()
    scores = torch.FloatTensor([[0.0, 0.0, 10.0, 20.0, 10.0]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # Compute result iteratively
    expected = 0.0
    for i in range(n[0]):
        for j in range(n[0]):
            inner = 1.0 / (1.0 + exp(
                -1.0 * float(scores[0, i] - scores[0, j])))
            inner = inner ** float(ys[0, i])
            expected -= log2(inner)

    assert loss.item() == approx(expected)


def test_lambda_arp1_worst():
    loss_fn = LambdaARPLoss1()
    scores = torch.FloatTensor([[4.0, 4.0, 2.0, 0.0, 2.0]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # Compute result iteratively
    expected = 0.0
    for i in range(n[0]):
        for j in range(n[0]):
            inner = 1.0 / (1.0 + exp(
                -1.0 * float(scores[0, i] - scores[0, j])))
            inner = inner ** float(ys[0, i])
            expected -= log2(inner)

    assert loss.item() == approx(expected)


def test_lambda_arp1_mid():
    loss_fn = LambdaARPLoss1()
    scores = torch.FloatTensor([[0.0, 1.0, 1.0, -2.0, 0.0]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([4])
    loss = loss_fn(scores, ys, n)

    # Compute result iteratively
    expected = 0.0
    for i in range(n[0]):
        for j in range(n[0]):
            inner = 1.0 / (1.0 + exp(
                -1.0 * float(scores[0, i] - scores[0, j])))
            inner = inner ** float(ys[0, i])
            expected -= log2(inner)

    assert loss.item() == approx(expected)


def test_lambda_arp2_perfect():
    loss_fn = LambdaARPLoss2()
    scores = torch.FloatTensor([[0.0, 0.0, 10.0, 20.0, 10.0]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # Compute result iteratively
    expected = 0.0
    for i in range(n[0]):
        for j in range(n[0]):
            if ys[0, i] > ys[0, j]:
                inner = 1.0 + exp(-1.0 * float(scores[0, i] - scores[0, j]))
                expected += abs(float(ys[0, i] - ys[0, j])) * log2(inner)

    assert loss.item() == approx(expected, rel=1e-06, abs=1e-6)


def test_lambda_arp2_worst():
    loss_fn = LambdaARPLoss2()
    scores = torch.FloatTensor([[4.0, 4.0, 2.0, 0.0, 2.0]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)

    # Compute result iteratively
    expected = 0.0
    for i in range(n[0]):
        for j in range(n[0]):
            if ys[0, i] > ys[0, j]:
                inner = 1.0 + exp(-1.0 * float(scores[0, i] - scores[0, j]))
                expected += abs(float(ys[0, i] - ys[0, j])) * log2(inner)

    assert loss.item() == approx(expected)


def test_lambda_arp2_mid():
    loss_fn = LambdaARPLoss2()
    scores = torch.FloatTensor([[0.0, 1.0, 1.0, -2.0, 0.0]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([4])
    loss = loss_fn(scores, ys, n)

    # Compute result iteratively
    expected = 0.0
    for i in range(n[0]):
        for j in range(n[0]):
            if ys[0, i] > ys[0, j]:
                inner = 1.0 + exp(-1.0 * float(scores[0, i] - scores[0, j]))
                expected += abs(float(ys[0, i] - ys[0, j])) * log2(inner)

    assert loss.item() == approx(expected)


def test_lambda_ndcg1_perfect():
    loss_fn = LambdaNDCGLoss1()
    scores = torch.FloatTensor([[0.0, 0.0, 10.0, 20.0, 10.0]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)
    sorting = [3, 4, 2, 0, 1]
    discounts = [log2(2.0 + i) for i in range(5)]
    max_dcg = (
        (2 ** 2.0 - 1.0) / log2(2.0) +
        (2 ** 1.0 - 1.0) / log2(3.0) +
        (2 ** 1.0 - 1.0) / log2(4.0))
    gains = [((2 ** float(ys[0, i])) - 1.0) / max_dcg
             for i in range(5)]

    # Compute result iteratively
    expected = 0.0
    for i in range(n[0]):
        for j in range(n[0]):
            si = sorting[i]
            sj = sorting[j]
            inner = 1.0 + exp(-1.0 * float(scores[0, si] - scores[0, sj]))
            inner = (1.0 / inner) ** (gains[si] / discounts[i])
            expected -= log2(inner)

    assert loss.item() == approx(expected)


def test_lambda_ndcg1_worst():
    loss_fn = LambdaNDCGLoss1()
    scores = torch.FloatTensor([[4.0, 4.0, 2.0, 0.0, 2.0]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)
    sorting = [1, 0, 2, 4, 3]
    discounts = [log2(2.0 + i) for i in range(5)]
    max_dcg = (
        (2 ** 2.0 - 1.0) / log2(2.0) +
        (2 ** 1.0 - 1.0) / log2(3.0) +
        (2 ** 1.0 - 1.0) / log2(4.0))
    gains = [((2 ** float(ys[0, i])) - 1.0) / max_dcg
             for i in range(5)]

    # Compute result iteratively
    expected = 0.0
    for i in range(n[0]):
        for j in range(n[0]):
            si = sorting[i]
            sj = sorting[j]
            inner = 1.0 + exp(-1.0 * float(scores[0, si] - scores[0, sj]))
            inner = (1.0 / inner) ** (gains[si] / discounts[i])
            expected -= log2(inner)

    assert loss.item() == approx(expected)


def test_lambda_ndcg1_mid():
    loss_fn = LambdaNDCGLoss1()
    scores = torch.FloatTensor([[0.0, 1.0, 1.5, -2.0, 0.0]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([4])
    loss = loss_fn(scores, ys, n)
    sorting = [2, 1, 0, 3, 4]
    discounts = [log2(2.0 + i) for i in range(5)]
    max_dcg = (2 ** 2.0 - 1.0) / log2(2.0) + (2 ** 1.0 - 1.0) / log2(3.0)
    gains = [((2 ** float(ys[0, i])) - 1.0) / max_dcg
             for i in range(5)]

    # Compute result iteratively
    expected = 0.0
    for i in range(n[0]):
        for j in range(n[0]):
            si = sorting[i]
            sj = sorting[j]
            inner = 1.0 + exp(-1.0 * float(scores[0, si] - scores[0, sj]))
            inner = (1.0 / inner) ** (gains[si] / discounts[i])
            expected -= log2(inner)

    assert loss.item() == approx(expected)


def test_lambda_ndcg2_perfect():
    loss_fn = LambdaNDCGLoss2()
    scores = torch.FloatTensor([[0.0, 0.0, 10.0, 20.0, 10.0]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)
    sorting = [3, 4, 2, 0, 1]
    discounts = [log2(2.0 + i) for i in range(5 + 1)]
    max_dcg = (
        (2 ** 2.0 - 1.0) / log2(2.0) +
        (2 ** 1.0 - 1.0) / log2(3.0) +
        (2 ** 1.0 - 1.0) / log2(4.0))
    gains = [((2 ** float(ys[0, i])) - 1.0) / max_dcg
             for i in range(5)]

    # Compute result iteratively
    expected = 0.0
    for i in range(n[0]):
        for j in range(n[0]):
            si = sorting[i]
            sj = sorting[j]
            if ys[0, si] > ys[0, sj]:
                score_diffs = float(scores[0, si] - scores[0, sj])
                inner = 1.0 / (1.0 + exp(-1.0 * score_diffs))
                delta_ij = abs((1.0 / discounts[abs(i - j)]) -
                               (1.0 / discounts[abs(i - j) + 1]))
                loss_pair = log2(
                    inner ** (delta_ij * abs(gains[si] - gains[sj])))
                expected -= loss_pair

    assert loss.item() == approx(expected, abs=1e-7)


def test_lambda_ndcg2_worst():
    loss_fn = LambdaNDCGLoss2()
    scores = torch.FloatTensor([[4.0, 4.0, 2.0, 0.0, 2.0]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([5])
    loss = loss_fn(scores, ys, n)
    sorting = [1, 0, 2, 4, 3]
    discounts = [log2(2.0 + i) for i in range(5 + 1)]
    max_dcg = (
        (2 ** 2.0 - 1.0) / log2(2.0) +
        (2 ** 1.0 - 1.0) / log2(3.0) +
        (2 ** 1.0 - 1.0) / log2(4.0))
    gains = [((2 ** float(ys[0, i])) - 1.0) / max_dcg
             for i in range(5)]

    # Compute result iteratively
    expected = 0.0
    for i in range(n[0]):
        for j in range(n[0]):
            si = sorting[i]
            sj = sorting[j]
            if ys[0, si] > ys[0, sj]:
                score_diffs = float(scores[0, si] - scores[0, sj])
                inner = 1.0 / (1.0 + exp(-1.0 * score_diffs))
                delta_ij = abs((1.0 / discounts[abs(i - j)]) -
                               (1.0 / discounts[abs(i - j) + 1]))
                loss_pair = log2(
                    inner ** (delta_ij * abs(gains[si] - gains[sj])))
                expected -= loss_pair

    assert loss.item() == approx(expected)


def test_lambda_ndcg2_mid():
    loss_fn = LambdaNDCGLoss2()
    scores = torch.FloatTensor([[0.0, 1.0, 1.5, -2.0, 0.0]])
    ys = torch.LongTensor([[0, 0, 1, 2, 1]])
    n = torch.LongTensor([4])
    loss = loss_fn(scores, ys, n)
    sorting = [2, 1, 0, 3, 4]
    discounts = [log2(2.0 + i) for i in range(5 + 1)]
    max_dcg = (2 ** 2.0 - 1.0) / log2(2.0) + (2 ** 1.0 - 1.0) / log2(3.0)
    gains = [((2 ** float(ys[0, i])) - 1.0) / max_dcg
             for i in range(5)]

    # Compute result iteratively
    expected = 0.0
    for i in range(n[0]):
        for j in range(n[0]):
            si = sorting[i]
            sj = sorting[j]
            if ys[0, si] > ys[0, sj]:
                score_diffs = float(scores[0, si] - scores[0, sj])
                inner = 1.0 / (1.0 + exp(-1.0 * score_diffs))
                delta_ij = abs((1.0 / discounts[abs(i - j)]) -
                               (1.0 / discounts[abs(i - j) + 1]))
                loss_pair = log2(
                    inner ** (delta_ij * abs(gains[si] - gains[sj])))
                expected -= loss_pair

    assert loss.item() == approx(expected)
