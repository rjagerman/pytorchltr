import torch
from pytorchltr.datasets.list_sampler import ListSampler
from pytorchltr.datasets.list_sampler import UniformSampler
from pytorchltr.datasets.list_sampler import BalancedRelevanceSampler

from pytest import approx


def rng(seed=1608637542):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def test_list_sampler():
    sampler = ListSampler(max_list_size=5)
    relevance = torch.tensor([0, 0, 1, 0, 0, 0, 2, 1], dtype=torch.long)
    idxs = sampler(relevance)
    expected = torch.arange(5)
    assert idxs.equal(expected)


def test_list_sampler_unlimited():
    sampler = ListSampler(max_list_size=None)
    relevance = torch.tensor([0, 0, 1, 0, 0, 0, 2, 1], dtype=torch.long)
    idxs = sampler(relevance)
    expected = torch.arange(8)
    assert idxs.equal(expected)


def test_list_sampler_single():
    sampler = ListSampler(max_list_size=1)
    relevance = torch.tensor([0], dtype=torch.long)
    idxs = sampler(relevance)
    expected = torch.arange(1)
    assert idxs.equal(expected)


def test_uniform_no_generator():
    torch.manual_seed(1608637542)
    sampler = UniformSampler(max_list_size=5)
    relevance = torch.tensor([0, 1, 0, 0, 2, 0, 1, 0, 0, 0], dtype=torch.long)
    idxs = sampler(relevance)
    assert idxs.shape == (5,)


def test_uniform_single():
    sampler = UniformSampler(max_list_size=1, generator=rng())
    relevance = torch.tensor([0], dtype=torch.long)
    idxs = sampler(relevance)
    expected = torch.tensor([0], dtype=torch.long)
    assert idxs.equal(expected)


def test_uniform_single_2():
    sampler = UniformSampler(max_list_size=1, generator=rng())
    relevance = torch.tensor([0, 1, 2, 0, 0, 1, 0, 0, 2], dtype=torch.long)
    idxs = sampler(relevance)
    assert idxs.shape == (1,)


def test_uniform_multiple():
    sampler = UniformSampler(max_list_size=3, generator=rng())
    relevance = torch.tensor([0, 0, 0, 1, 0, 0, 2], dtype=torch.long)
    idxs = sampler(relevance)
    assert idxs.shape == (3,)


def test_uniform_multiple_2():
    sampler = UniformSampler(max_list_size=9, generator=rng())
    relevance = torch.tensor([0, 1], dtype=torch.long)
    idxs = sampler(relevance)
    assert idxs.shape == (2,)


def test_uniform_unlimited():
    sampler = UniformSampler(max_list_size=None, generator=rng())
    relevance = torch.tensor([0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 1],
                             dtype=torch.long)
    idxs = sampler(relevance)
    assert idxs.shape == (13,)


def test_uniform_stat():
    sampler = UniformSampler(max_list_size=1, generator=rng())
    relevance = torch.tensor([0, 0, 1, 0, 0, 0, 2, 1], dtype=torch.long)
    n = 1000
    hist = torch.zeros(3)
    for i in range(n):
        idxs = sampler(relevance)
        hist[relevance[idxs].item()] += 1
    hist /= n
    assert hist[0].item() == approx(5.0 / 8.0, abs=0.05)
    assert hist[1].item() == approx(2.0 / 8.0, abs=0.05)
    assert hist[2].item() == approx(1.0 / 8.0, abs=0.05)


def test_uniform_stat_large():
    sampler = UniformSampler(max_list_size=5, generator=rng())
    relevance = torch.tensor([0, 0, 1, 0, 0, 0, 2, 1], dtype=torch.long)
    n = 1000
    for idx in range(5):
        hist = torch.zeros(3)
        for i in range(n):
            idxs = sampler(relevance)
            hist[relevance[idxs][idx].item()] += 1
        hist /= n
        assert hist[0].item() == approx(5.0 / 8.0, abs=0.05)
        assert hist[1].item() == approx(2.0 / 8.0, abs=0.05)
        assert hist[2].item() == approx(1.0 / 8.0, abs=0.05)


def test_balanced_no_generator():
    torch.manual_seed(1608637542)
    sampler = BalancedRelevanceSampler(max_list_size=5)
    relevance = torch.tensor([0, 1, 0, 0, 2, 0, 1, 0, 0, 0], dtype=torch.long)
    idxs = sampler(relevance)
    assert idxs.shape == (5,)


def test_balanced_single():
    sampler = BalancedRelevanceSampler(max_list_size=1, generator=rng())
    relevance = torch.tensor([0], dtype=torch.long)
    idxs = sampler(relevance)
    expected = torch.tensor([0], dtype=torch.long)
    assert idxs.equal(expected)


def test_balanced_single_2():
    sampler = BalancedRelevanceSampler(max_list_size=1, generator=rng())
    relevance = torch.tensor([0, 1, 2, 0, 0, 1, 0, 0, 2], dtype=torch.long)
    idxs = sampler(relevance)
    assert idxs.shape == (1,)


def test_balanced_multiple():
    sampler = BalancedRelevanceSampler(max_list_size=3, generator=rng())
    relevance = torch.tensor([0, 0, 0, 1, 0, 0, 2], dtype=torch.long)
    idxs = sampler(relevance)
    assert idxs.shape == (3,)


def test_balanced_multiple_2():
    sampler = BalancedRelevanceSampler(max_list_size=9, generator=rng())
    relevance = torch.tensor([0, 1], dtype=torch.long)
    idxs = sampler(relevance)
    assert idxs.shape == (2,)


def test_balanced_unlimited():
    sampler = BalancedRelevanceSampler(max_list_size=None, generator=rng())
    relevance = torch.tensor([0, 0, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 1],
                             dtype=torch.long)
    idxs = sampler(relevance)
    assert idxs.shape == (13,)


def test_balanced_stat():
    sampler = BalancedRelevanceSampler(max_list_size=1, generator=rng())
    relevance = torch.tensor([0, 0, 1, 0, 0, 0, 2, 1], dtype=torch.long)
    n = 1000
    hist = torch.zeros(3)
    for i in range(n):
        idxs = sampler(relevance)
        hist[relevance[idxs].item()] += 1
    hist /= n
    assert hist[0].item() == approx(1.0 / 3.0, abs=0.05)
    assert hist[1].item() == approx(1.0 / 3.0, abs=0.05)
    assert hist[2].item() == approx(1.0 / 3.0, abs=0.05)


def test_balanced_stat_large():
    sampler = BalancedRelevanceSampler(max_list_size=7, generator=rng())
    relevance = torch.tensor([0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0],
                             dtype=torch.long)
    n = 1000
    expected = torch.tensor([
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [1.0 / 2.0, 1.0 / 2.0, 0.0],
        [1.0 / 2.0, 1.0 / 2.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    for idx in range(7):
        hist = torch.zeros(3)
        for i in range(n):
            idxs = sampler(relevance)
            hist[relevance[idxs][idx].item()] += 1
        hist /= n
        assert hist[0].item() == approx(expected[idx, 0].item(), abs=0.05)
        assert hist[1].item() == approx(expected[idx, 1].item(), abs=0.05)
        assert hist[2].item() == approx(expected[idx, 2].item(), abs=0.05)
