import torch
from pytorchltr.click_simulation import simulate_perfect
from pytorchltr.click_simulation import simulate_position
from pytorchltr.click_simulation import simulate_nearrandom
from pytest import approx


def _generate_test_data():
    """Generates a small batch of test data for simulations."""
    rankings = torch.LongTensor([
        [3, 4, 0, 2, 1],
        [1, 0, 2, 4, 3]
    ])
    ys = torch.LongTensor([
        [1, 0, 4, 0, 2],
        [4, 3, 0, 0, 0]
    ])
    n = torch.LongTensor([5, 3])
    return rankings, ys, n


def _monte_carlo_simulation(rankings, ys, n, click_fn, nr=100):
    """Runs a click simulation repeatedly and reports averages.

    Arguments:
        rankings: The batch of rankings.
        ys: The batch of relevance labels.
        n: The nr of docs per query in the batch.
        click_fn: The click simulate function to call.
        nr: Number of simulations to run.

    Returns:
        The averaged output of `click_fn` across `nr` runs.
    """
    clicks = simulate_perfect(rankings, ys, n)
    torch.manual_seed(4200)
    click_aggregate = torch.zeros_like(clicks[0]).to(torch.float)
    prop_aggregate = torch.zeros_like(clicks[1]).to(torch.float)
    for i in range(nr):
        clicks, props = click_fn(rankings, ys, n)
        click_aggregate += clicks.to(dtype=torch.float)
        prop_aggregate += props
    return click_aggregate / float(nr), prop_aggregate / float(nr)


def test_perfect_clicks():
    rankings, ys, n = _generate_test_data()
    clicks, props = _monte_carlo_simulation(rankings, ys, n, simulate_perfect)
    rel_expected = torch.FloatTensor([
        [0.2, 0.0, 1.0, 0.0, 0.4],
        [1.0, 0.8, 0.0, 0.0, 0.0]
    ])
    props_expected = torch.FloatTensor([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.0]
    ])
    clicks_expected = rel_expected * props_expected
    assert clicks_expected.numpy() == approx(clicks.numpy(), abs=0.1)
    assert props_expected.numpy() == approx(props.numpy(), abs=0.1)


def test_perfect_clicks_cutoff_3():
    rankings, ys, n = _generate_test_data()

    def click_fn(rankings, ys, n):
        return simulate_perfect(rankings, ys, n, cutoff=3)
    clicks, props = _monte_carlo_simulation(rankings, ys, n, click_fn)
    rel_expected = torch.FloatTensor([
        [0.2, 0.0, 1.0, 0.0, 0.4],
        [1.0, 0.8, 0.0, 0.0, 0.0]
    ])
    props_expected = torch.FloatTensor([
        [1.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.0]
    ])
    clicks_expected = rel_expected * props_expected
    assert clicks_expected.numpy() == approx(clicks.numpy(), abs=0.1)
    assert props_expected.numpy() == approx(props.numpy(), abs=0.1)


def test_perfect_clicks_cutoff_2():
    rankings, ys, n = _generate_test_data()

    def click_fn(rankings, ys, n):
        return simulate_perfect(rankings, ys, n, cutoff=2)
    clicks, props = _monte_carlo_simulation(rankings, ys, n, click_fn)
    rel_expected = torch.FloatTensor([
        [0.2, 0.0, 1.0, 0.0, 0.4],
        [1.0, 0.8, 0.0, 0.0, 0.0]
    ])
    props_expected = torch.FloatTensor([
        [0.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.0, 0.0]
    ])
    clicks_expected = rel_expected * props_expected
    assert clicks_expected.numpy() == approx(clicks.numpy(), abs=0.1)
    assert props_expected.numpy() == approx(props.numpy(), abs=0.1)


def test_position_clicks():
    rankings, ys, n = _generate_test_data()
    clicks, props = _monte_carlo_simulation(rankings, ys, n, simulate_position)
    rel_expected = torch.FloatTensor([
        [0.1, 0.1, 1.0, 0.1, 0.1],
        [1.0, 1.0, 0.1, 0.1, 0.1]
    ])
    props_expected = torch.FloatTensor([
        [1/4.0, 1/6.0, 1/5.0, 1/2.0, 1/3.0],
        [1/3.0, 1/2.0, 1/4.0, 0.0, 0.0]
    ])
    clicks_expected = rel_expected * props_expected
    assert clicks_expected.numpy() == approx(clicks.numpy(), abs=0.1)
    assert props_expected.numpy() == approx(props.numpy(), abs=0.1)


def test_position_clicks_eta_2():
    rankings, ys, n = _generate_test_data()

    def click_fn(rankings, ys, n):
        return simulate_position(rankings, ys, n, eta=2.0)
    clicks, props = _monte_carlo_simulation(rankings, ys, n, click_fn)
    rel_expected = torch.FloatTensor([
        [0.1, 0.1, 1.0, 0.1, 0.1],
        [1.0, 1.0, 0.1, 0.1, 0.1]
    ])
    props_expected = torch.FloatTensor([
        [1/4.0, 1/6.0, 1/5.0, 1/2.0, 1/3.0],
        [1/3.0, 1/2.0, 1/4.0, 0.0, 0.0]
    ]) ** 2.0
    clicks_expected = rel_expected * props_expected
    assert clicks_expected.numpy() == approx(clicks.numpy(), abs=0.1)
    assert props_expected.numpy() == approx(props.numpy(), abs=0.1)


def test_position_clicks_eta_0():
    rankings, ys, n = _generate_test_data()

    def click_fn(rankings, ys, n):
        return simulate_position(rankings, ys, n, eta=0.0)
    clicks, props = _monte_carlo_simulation(rankings, ys, n, click_fn)
    rel_expected = torch.FloatTensor([
        [0.1, 0.1, 1.0, 0.1, 0.1],
        [1.0, 1.0, 0.1, 0.1, 0.1]
    ])
    props_expected = torch.FloatTensor([
        [1/4.0, 1/6.0, 1/5.0, 1/2.0, 1/3.0],
        [1/3.0, 1/2.0, 1/4.0, 0.0, 0.0]
    ]) ** 0.0
    props_expected[1, 3] = 0.0
    props_expected[1, 4] = 0.0
    clicks_expected = rel_expected * props_expected
    assert clicks_expected.numpy() == approx(clicks.numpy(), abs=0.1)
    assert props_expected.numpy() == approx(props.numpy(), abs=0.1)


def test_position_clicks_cutoff_3():
    rankings, ys, n = _generate_test_data()

    def click_fn(rankings, ys, n):
        return simulate_position(rankings, ys, n, cutoff=3)
    clicks, props = _monte_carlo_simulation(rankings, ys, n, click_fn)
    rel_expected = torch.FloatTensor([
        [0.1, 0.1, 1.0, 0.1, 0.1],
        [1.0, 1.0, 0.1, 0.1, 0.1]
    ])
    props_expected = torch.FloatTensor([
        [1/4.0, 0.0, 0.0, 1/2.0, 1/3.0],
        [1/3.0, 1/2.0, 1/4.0, 0.0, 0.0]
    ])
    clicks_expected = rel_expected * props_expected
    assert clicks_expected.numpy() == approx(clicks.numpy(), abs=0.1)
    assert props_expected.numpy() == approx(props.numpy(), abs=0.1)


def test_nearrandom_clicks():
    rankings, ys, n = _generate_test_data()
    clicks, props = _monte_carlo_simulation(
        rankings, ys, n, simulate_nearrandom)
    rel_expected = torch.FloatTensor([
        [0.45, 0.4, 0.6, 0.4, 0.5],
        [0.6, 0.55, 0.4, 0.4, 0.4]
    ])
    props_expected = torch.FloatTensor([
        [1/4.0, 1/6.0, 1/5.0, 1/2.0, 1/3.0],
        [1/3.0, 1/2.0, 1/4.0, 0.0, 0.0]
    ])
    clicks_expected = rel_expected * props_expected
    assert clicks_expected.numpy() == approx(clicks.numpy(), abs=0.1)
    assert props_expected.numpy() == approx(props.numpy(), abs=0.1)


def test_nearrandom_clicks_eta_2():
    rankings, ys, n = _generate_test_data()

    def click_fn(rankings, ys, n):
        return simulate_nearrandom(rankings, ys, n, eta=2.0)
    clicks, props = _monte_carlo_simulation(rankings, ys, n, click_fn)
    rel_expected = torch.FloatTensor([
        [0.45, 0.4, 0.6, 0.4, 0.5],
        [0.6, 0.55, 0.4, 0.4, 0.4]
    ])
    props_expected = torch.FloatTensor([
        [1/4.0, 1/6.0, 1/5.0, 1/2.0, 1/3.0],
        [1/3.0, 1/2.0, 1/4.0, 0.0, 0.0]
    ]) ** 2.0
    clicks_expected = rel_expected * props_expected
    assert clicks_expected.numpy() == approx(clicks.numpy(), abs=0.1)
    assert props_expected.numpy() == approx(props.numpy(), abs=0.1)


def test_nearrandom_clicks_eta_0():
    rankings, ys, n = _generate_test_data()

    def click_fn(rankings, ys, n):
        return simulate_nearrandom(rankings, ys, n, eta=0.0)
    clicks, props = _monte_carlo_simulation(rankings, ys, n, click_fn)
    rel_expected = torch.FloatTensor([
        [0.45, 0.4, 0.6, 0.4, 0.5],
        [0.6, 0.55, 0.4, 0.4, 0.4]
    ])
    props_expected = torch.FloatTensor([
        [1/4.0, 1/6.0, 1/5.0, 1/2.0, 1/3.0],
        [1/3.0, 1/2.0, 1/4.0, 0.0, 0.0]
    ]) ** 0.0
    props_expected[1, 3] = 0.0
    props_expected[1, 4] = 0.0
    clicks_expected = rel_expected * props_expected
    assert clicks_expected.numpy() == approx(clicks.numpy(), abs=0.1)
    assert props_expected.numpy() == approx(props.numpy(), abs=0.1)


def test_nearrandom_clicks_cutoff_3():
    rankings, ys, n = _generate_test_data()

    def click_fn(rankings, ys, n):
        return simulate_nearrandom(rankings, ys, n, cutoff=3)
    clicks, props = _monte_carlo_simulation(rankings, ys, n, click_fn)
    rel_expected = torch.FloatTensor([
        [0.45, 0.4, 0.6, 0.4, 0.5],
        [0.6, 0.55, 0.4, 0.4, 0.4]
    ])
    props_expected = torch.FloatTensor([
        [1/4.0, 0.0, 0.0, 1/2.0, 1/3.0],
        [1/3.0, 1/2.0, 1/4.0, 0.0, 0.0]
    ])
    clicks_expected = rel_expected * props_expected
    assert clicks_expected.numpy() == approx(clicks.numpy(), abs=0.1)
    assert props_expected.numpy() == approx(props.numpy(), abs=0.1)
