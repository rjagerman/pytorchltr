import torch
from pytorchltr.evaluation.trec import generate_pytrec_eval


def test_pytrec_eval_input():
    scores = torch.FloatTensor([
        [10.0, 5.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 4.0, 2.0, 5.5]
    ])
    ys = torch.LongTensor([
        [0, 1, 1, 0, 1],
        [3, 1, 0, 1, 0]
    ])
    n = torch.LongTensor([5, 4])

    qrels, run = generate_pytrec_eval(scores, ys, n)
    qrels_expected = {
        'q0': {'d0': 0, 'd1': 1, 'd2': 1, 'd3': 0, 'd4': 1},
        'q1': {'d0': 3, 'd1': 1, 'd2': 0, 'd3': 1}
    }
    run_expected = {
        'q0': {'d0': 10.0, 'd1': 5.0, 'd2': 2.0, 'd3': 3.0, 'd4': 4.0},
        'q1': {'d0': 5.0, 'd1': 6.0, 'd2': 4.0, 'd3': 2.0}
    }
    assert qrels_expected == qrels
    assert run_expected == run


def test_pytrec_eval_input_noprefix():
    scores = torch.FloatTensor([
        [10.0, 5.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 4.0, 2.0, 5.5]
    ])
    ys = torch.LongTensor([
        [0, 1, 1, 0, 1],
        [3, 1, 0, 1, 0]
    ])
    n = torch.LongTensor([5, 4])

    qrels, run = generate_pytrec_eval(scores, ys, n, q_prefix="", d_prefix="")
    qrels_expected = {
        '0': {'0': 0, '1': 1, '2': 1, '3': 0, '4': 1},
        '1': {'0': 3, '1': 1, '2': 0, '3': 1}
    }
    run_expected = {
        '0': {'0': 10.0, '1': 5.0, '2': 2.0, '3': 3.0, '4': 4.0},
        '1': {'0': 5.0, '1': 6.0, '2': 4.0, '3': 2.0}
    }
    assert qrels_expected == qrels
    assert run_expected == run


def test_pytrec_eval_input_qids():
    scores = torch.FloatTensor([
        [10.0, 5.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 4.0, 2.0, 5.5]
    ])
    ys = torch.LongTensor([
        [0, 1, 1, 0, 1],
        [3, 1, 0, 1, 0]
    ])
    n = torch.LongTensor([5, 4])
    qid = torch.LongTensor([15623, 49998])

    qrels, run = generate_pytrec_eval(scores, ys, n, qid, q_prefix="")
    qrels_expected = {
        '15623': {'d0': 0, 'd1': 1, 'd2': 1, 'd3': 0, 'd4': 1},
        '49998': {'d0': 3, 'd1': 1, 'd2': 0, 'd3': 1}
    }
    run_expected = {
        '15623': {'d0': 10.0, 'd1': 5.0, 'd2': 2.0, 'd3': 3.0, 'd4': 4.0},
        '49998': {'d0': 5.0, 'd1': 6.0, 'd2': 4.0, 'd3': 2.0}
    }
    assert qrels_expected == qrels
    assert run_expected == run
