import torch
from pytorchltr.evaluation import dcg
from pytorchltr.evaluation import ndcg
from pytest import approx


def _generate_data():
    scores = torch.FloatTensor([
        [10.0, 5.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 4.0, 2.0, 5.5]
    ])
    ys = torch.LongTensor([
        [0, 1, 1, 0, 1],
        [3, 1, 0, 1, 0]
    ])
    n = torch.LongTensor([5, 4])
    return scores, ys, n


def test_dcg3():
    torch.manual_seed(42)
    scores, ys, n = _generate_data()
    out = dcg(scores, ys, n, k=3, exp=True)

    expected = torch.FloatTensor([
        1.1309297535714575,
        5.4165082750002025])

    assert out.numpy() == approx(expected.numpy())


def test_ndcg3():
    torch.manual_seed(42)
    scores, ys, n = _generate_data()
    out = ndcg(scores, ys, n, k=3, exp=True)

    expected = torch.FloatTensor([
        1.1309297535714575 / 2.1309297535714578,
        5.4165082750002025 / 8.130929753571458])

    assert out.numpy() == approx(expected.numpy())


def test_dcg5_exp():
    torch.manual_seed(42)
    scores, ys, n = _generate_data()
    out = dcg(scores, ys, n, k=5, exp=True)

    expected = torch.FloatTensor([
        1.5177825608059992,
        5.847184833073595])

    assert out.numpy() == approx(expected.numpy())


def test_ndcg5_exp():
    torch.manual_seed(42)
    scores, ys, n = _generate_data()
    out = ndcg(scores, ys, n, k=5, exp=True)

    expected = torch.FloatTensor([
        1.5177825608059992 / 2.1309297535714578,
        5.847184833073595 / 8.130929753571458])

    assert out.numpy() == approx(expected.numpy())


def test_dcg5_nonexp():
    torch.manual_seed(42)
    scores, ys, n = _generate_data()
    out = dcg(scores, ys, n, k=5, exp=False)

    expected = torch.FloatTensor([
        1.5177825608059992,
        3.3234658187877653])

    assert out.numpy() == approx(expected.numpy())


def test_ndcg5_nonexp():
    torch.manual_seed(42)
    scores, ys, n = _generate_data()
    out = ndcg(scores, ys, n, k=5, exp=False)

    expected = torch.FloatTensor([
        1.5177825608059992 / 2.1309297535714578,
        3.3234658187877653 / 4.130929753571458])

    assert out.numpy() == approx(expected.numpy())


def test_dcg_all_relevant():
    torch.manual_seed(42)
    scores = torch.FloatTensor([
        [10.0, 5.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 4.0, 2.0, 5.5]
    ])
    ys = torch.LongTensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ])
    n = torch.LongTensor([5, 4])
    out = dcg(scores, ys, n, k=5, exp=False)

    expected = torch.sum(torch.repeat_interleave(
        1.0 / torch.log2(2.0 + torch.arange(5, dtype=torch.float))[None, :],
        2, dim=0), dim=1)

    assert out.numpy() == approx(expected.numpy())


def test_ndcg_all_relevant():
    torch.manual_seed(42)
    scores = torch.FloatTensor([
        [10.0, 5.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 4.0, 2.0, 5.5]
    ])
    ys = torch.LongTensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ])
    n = torch.LongTensor([5, 4])
    out = ndcg(scores, ys, n, k=5, exp=False)
    expected = torch.ones(2)

    assert out.numpy() == approx(expected.numpy())


def test_dcg_no_relevant():
    torch.manual_seed(42)
    scores = torch.FloatTensor([
        [10.0, 5.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 4.0, 2.0, 5.5]
    ])
    ys = torch.LongTensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    n = torch.LongTensor([5, 4])

    out = dcg(scores, ys, n, k=5, exp=False)
    expected = torch.zeros(2)

    assert out.numpy() == approx(expected.numpy())


def test_ndcg_no_relevant():
    torch.manual_seed(42)
    scores = torch.FloatTensor([
        [10.0, 5.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 4.0, 2.0, 5.5]
    ])
    ys = torch.LongTensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    n = torch.LongTensor([5, 4])

    out = ndcg(scores, ys, n, k=5, exp=False)
    expected = torch.zeros(2)

    assert out.numpy() == approx(expected.numpy())
