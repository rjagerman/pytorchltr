import torch
from pytorchltr.utils.tensor_operations import rank_by_plackettluce
from pytest import approx


def repeat_rank_stats(fn, runs=100):
    """Repeats a ranking function multiple times and records how often each
    item gets placed at each rank.

    Args:
        fn: The function that generates rankings.
        runs: How many runs to repeat.
    """
    shape = fn().shape
    out = torch.zeros((shape[0], shape[1], shape[1]))
    for i in range(runs):
        ranking = fn()
        for batch in range(shape[0]):
            for j, r_j in enumerate(ranking[batch, :]):
                out[batch, j, r_j] += 1
    return out / runs


def test_plackettluce_3d_input():
    torch.manual_seed(42)
    scores = torch.FloatTensor([[[5.0], [3.0], [2.0]]])
    n = torch.IntTensor([3])
    fn = lambda: rank_by_plackettluce(scores, n)  # noqa: E731
    out = repeat_rank_stats(fn, runs=100)
    expected = torch.nn.Softmax(dim=1)(scores.reshape((1, 3)))

    # Assert that rank 1 has a proportional amount of each doc
    out = out[0, 0, :].numpy()
    expected = expected[0, :].numpy()
    assert out == approx(expected, abs=0.1)


def test_plackettluce_rank_1():
    torch.manual_seed(42)
    scores = torch.FloatTensor([[5.0, 3.0, 2.0, 1.0]])
    n = torch.IntTensor([4])
    fn = lambda: rank_by_plackettluce(scores, n)  # noqa: E731
    out = repeat_rank_stats(fn, runs=100)
    expected = torch.nn.Softmax(dim=1)(scores)

    # Assert that rank 1 has a proportional amount of each doc
    out = out[0, 0, :].numpy()
    expected = expected[0, :].numpy()
    assert out == approx(expected, abs=0.1)


def test_plackettluce_rank_2():
    torch.manual_seed(42)
    scores = torch.FloatTensor([[5.0, 3.0, 2.0, 1.0]])
    n = torch.IntTensor([4])
    fn = lambda: rank_by_plackettluce(scores, n)  # noqa: E731
    out = repeat_rank_stats(fn, runs=100)
    softmax = torch.nn.Softmax(dim=1)(scores)

    # Compute:
    # P(doc in rank 2) =
    #    sum_{i} P(doc_i in rank 1) * P(doc in rank 2 | doc_i in rank 1)
    expected = torch.zeros(softmax.shape)
    for i in range(4):
        # P(doc_i in rank1):
        p_rank_1 = softmax[0, i]

        # P(doc in rank 2 | doc_i in rank 1):
        r_scores = scores.clone()
        r_scores[scores == scores[0, i]] = 0.0
        r_softmax = torch.exp(r_scores) / torch.sum(torch.exp(r_scores))

        # P(doc in rank 2):
        expected += p_rank_1 * r_softmax

    # Assert that rank 2 has a proportional amount of each doc
    out = out[0, 1, :].numpy()
    expected = expected[0, :].numpy()
    assert out == approx(expected, abs=0.1)


def test_plackettluce_place_padded_docs_last():
    torch.manual_seed(42)
    scores = torch.FloatTensor([[5.0, 3.0, 2.0, 1.0, 10.0]])
    n = torch.IntTensor([4])
    fn = lambda: rank_by_plackettluce(scores, n)  # noqa: E731
    out = repeat_rank_stats(fn, runs=100)

    # Assert that 4th doc is placed last always
    assert out[0, 4, 0].numpy() == approx(0.0)
    assert out[0, 4, 1].numpy() == approx(0.0)
    assert out[0, 4, 2].numpy() == approx(0.0)
    assert out[0, 4, 3].numpy() == approx(0.0)
    assert out[0, 4, 4].numpy() == approx(1.0)


def test_plackettluce_batch():
    torch.manual_seed(42)
    scores = torch.FloatTensor([[5.0, 3.0, 2.0, 1.0],
                                [10.0, 3.0, 10.0, 100.0]])
    n = torch.IntTensor([4, 4])
    fn = lambda: rank_by_plackettluce(scores, n)  # noqa: E731
    out = repeat_rank_stats(fn, runs=100)
    expected = torch.nn.Softmax(dim=1)(scores)

    # Assert that both rows in the batch are correct
    out_1 = out[0, 0, :].numpy()
    expected_1 = expected[0, :].numpy()
    assert out_1 == approx(expected_1, abs=0.1)

    out_2 = out[1, 0, :].numpy()
    expected_2 = expected[1, :].numpy()
    assert out_2 == approx(expected_2, abs=0.1)


def test_plackettluce_negative_input():
    torch.manual_seed(42)
    scores = torch.FloatTensor([[-1.0, -2.0, 0.0]])
    n = torch.IntTensor([3])
    fn = lambda: rank_by_plackettluce(scores, n)  # noqa: E731
    out = repeat_rank_stats(fn, runs=100)
    expected = torch.nn.Softmax(dim=1)(scores)

    # Assert that rank 1 has a proportional amount of each doc
    out = out[0, 0, :].numpy()
    expected = expected[0, :].numpy()
    assert out == approx(expected, abs=0.1)
