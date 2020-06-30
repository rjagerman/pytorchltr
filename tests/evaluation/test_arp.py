import torch
from pytorchltr.evaluation import arp
from pytest import approx


def test_arp():
    torch.manual_seed(42)
    scores = torch.FloatTensor([
        [10.0, 5.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 4.0, 2.0, 5.5]
    ])
    ys = torch.LongTensor([
        [0, 1, 1, 0, 1],
        [1, 1, 0, 0, 0]
    ])
    n = torch.LongTensor([5, 4])

    out = arp(scores, ys, n)
    expected = torch.FloatTensor([3.333333333, 1.5])

    assert out.numpy() == approx(expected.numpy())


def test_arp_all_relevant():
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

    out = arp(scores, ys, n)
    expected = torch.FloatTensor([3.0, 2.5])

    assert out.numpy() == approx(expected.numpy())


def test_arp_no_relevant():
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

    out = arp(scores, ys, n)
    expected = torch.FloatTensor([0.0, 0.0])

    assert out.numpy() == approx(expected.numpy())
