"""Generate pytrec_eval runs from model and labels."""


def generate_pytrec_eval(scores, relevance, n, qids=None, qid_offset=0,
                         q_prefix="q", d_prefix="d"):
    """Generates pytrec_eval qrels and runs from given batch.

    Arguments:
        scores: A FloatTensor of size (batch_size, list_size) indicating the
            scores of each document.
        relevance: A LongTensor of size (batch_size, list_size) indicating the
            relevance of each document.
        n: A LongTensor of size (batch_size) indicating the number of docs per
            query.
        qids: (Optional) a LongTensor of size (batch_size) indicating the qid
            of each query.
        qid_offset: (Optional) an offset to increment all qids in this batch
            with. Only used if `qids` is None.
        q_prefix: (Optional) a string prefix to add for query identifiers.
        d_prefix: (Optional) a string prefix to add for doc identifiers.

    Returns:
        A tuple of dicts containing a qrel dict and a run dict.
    """
    qrel = {}
    run = {}
    for i in range(scores.shape[0]):

        # Store under correct qid.
        if qids is not None:
            qid = f"{q_prefix}{int(qids[i])}"
        else:
            qid = f"{q_prefix}{i + qid_offset}"
        qrel[qid] = {}
        run[qid] = {}

        # Iterate documents and get relevance and scores.
        for d in range(n[i]):
            qrel[qid][f"{d_prefix}{d}"] = int(relevance[i, d])
            run[qid][f"{d_prefix}{d}"] = float(scores[i, d])

    return qrel, run
