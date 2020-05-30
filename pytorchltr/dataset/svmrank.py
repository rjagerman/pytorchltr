"""Data loading for SVMRank-style data sets."""
from torch.utils.data import Dataset as _Dataset
from sklearn.datasets import load_svmlight_file as _load_svmlight_file
from scipy.sparse import coo_matrix as _coo_matrix
import numpy as _np
import torch as _torch


class SVMRankingDataset(_Dataset):
    def __init__(self, xs, ys, unique_qids, offsets, sparse):
        """An SVM ranking dataset supporting both dense and sparse tensors.

        Call `svmranking_dataset()` instead of constructing this class
        directly.

        Arguments:
            xs (scipy.sparse.csr_matrix or np.ndarray): The query-document
                feature vectors. Should be a scipy.sparse.csr_matrix if sparse
                is True, otherwise an np.ndarray
            ys (np.ndarray): The per query-document relevance labels.
            unique_qids (np.ndarray): The unique query identifiers per query.
            offsets (np.ndarray): The offsets for each query in xs and ys.
            sparse (bool): True if xs is a sparse matrix, False otherwise.
        """
        self._xs = xs
        self._ys = ys
        self._unique_qids = unique_qids
        self._qid_map = {qid: index for index, qid in enumerate(unique_qids)}
        self._offsets = offsets
        self._n = len(self._offsets) - 1
        self._sparse = sparse

    def get_index(self, qid):
        return self._qid_map[qid]

    def __getitem__(self, index):
        # Extract query features and relevance labels
        qid = self._unique_qids[index]
        start = self._offsets[index]
        end = self._offsets[index + 1]
        features = self._xs[start:end, :]
        y = _torch.LongTensor(self._ys[start:end])
        n = end - start

        # Compute sparse or dense torch tensor
        if self._sparse:
            coo = _coo_matrix(features)
            ind = _torch.LongTensor(_np.vstack((coo.row, coo.col)))
            val = _torch.FloatTensor(coo.data)
            features = _torch.sparse.FloatTensor(
                ind, val, _torch.Size(coo.shape))
        else:
            features = _torch.FloatTensor(features)

        # Return data sample
        return {
            "features": features,
            "relevance": y,
            "qid": qid,
            "n": n,
            "sparse": self._sparse
        }

    def __len__(self):
        return self._n


def svmranking_dataset(file, sparse=False, normalize=False,
                       filter_queries=False, zero_based="auto"):
    """Loads an SVMRank-style dataset from given file_path.

    Arguments:
        file (str or file-like): The path to load the dataset from.
        sparse (bool, optional): Whether to load the features as sparse
            features.
        normalize (bool, optional): Whether to perform query-level
            normalization (requires non-sparse features).
        zero_based (str or int, optional): The zero based index.
    """
    # Load svmlight file
    xs, ys, qids = _load_svmlight_file(
        file, query_id=True, zero_based=zero_based)

    # Compute query offsets and unique qids
    offsets = _np.hstack(
        [[0], _np.where(qids[1:] != qids[:-1])[0] + 1, [len(qids)]])
    unique_qids = qids[offsets[:-1]]

    # Densify
    if not sparse:
        xs = xs.A

    # Normalize xs
    if normalize:
        if sparse:
            raise NotImplementedError(
                "Normalization without dense features is not supported.")
        _per_offset_normalize(xs, offsets)

    # Create full dataset
    dataset = SVMRankingDataset(xs, ys, unique_qids, offsets, sparse)

    # Filter out queries with only non-relevant reuslts
    if filter_queries:
        indices = []
        for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            if _np.sum(ys[start:end]) > 0.0:
                indices.append(i)
        indices = _np.array(indices)
        dataset = _torch.utils.data.Subset(dataset, indices)

    return dataset


def _per_offset_normalize(xs, offsets):
    """Performs query-level normalization using xs and offsets."""
    for start, end in zip(offsets[:-1], offsets[1:]):
        xs[start:end, :] -= _np.min(xs[start:end, :], axis=0)
        m = _np.max(xs[start:end, :], axis=0)
        m[m == 0.0] = 1.0
        xs[start:end, :] /= m


def create_svmranking_collate_fn(max_list_size=None,
                                 rng=None):
    """Creates a collate function for batches of svm rank examples.

    Arguments:
        max_list_size: The maximum list size (cuts off documents beyond this).
        rng: A numpy random state for selecting indices to cut off.
    """
    if rng is None:
        rng = _np.random.RandomState(42)

    def _collate_fn(batch):
        # Check if batch is sparse or not
        sparse = batch[0]["sparse"]

        # Compute list size
        list_size = max([b['features'].shape[0] for b in batch])
        if max_list_size is not None:
            list_size = min(max_list_size, list_size)

        # Create output tensors from batch
        if sparse:
            out_features = []
        else:
            out_features = _torch.zeros(
                (len(batch), list_size, batch[0]['features'].shape[1]))
        out_relevance = _torch.zeros(
            (len(batch), list_size), dtype=_torch.long)
        out_qid = _torch.zeros(len(batch), dtype=_torch.long)
        out_n = _torch.zeros(len(batch), dtype=_torch.long)

        # Collate the whole batch
        for batch_index, sample in enumerate(batch):

            # Generate random indices when we exceed the list_size.
            xs = sample["features"]
            if xs.shape[0] > list_size:
                rng_indices = _np.sort(
                    rng.permutation(xs.shape[0])[:list_size])

            # Collate features
            if sparse:
                xs_coalesce = xs.coalesce()
                ind = xs_coalesce.indices()
                val = xs_coalesce.values()
                if xs.shape[0] > list_size:
                    mask = [ind[0, :] == i for i in rng_indices]
                    for i in range(len(mask)):
                        ind[0, mask[i]] = int(i)
                    ind = ind[:, sum(mask)]
                    val = val[sum(mask)]
                ind_l = _torch.ones((1, ind.shape[1]),
                                    dtype=ind.dtype) * batch_index
                ind = _torch.cat([ind_l, ind], dim=0)
                out_features.append((ind, val))
            else:
                if xs.shape[0] > list_size:
                    out_features[batch_index, :, :] = xs[rng_indices, :]
                else:
                    out_features[batch_index, 0:xs.shape[0], :] = xs

            # Collate relevance
            if xs.shape[0] > list_size:
                out_relevance[batch_index, 0:len(rng_indices)] = sample["relevance"][rng_indices]  # noqa: E501
            else:
                out_relevance[batch_index, 0:len(sample["relevance"])] = sample["relevance"]  # noqa: E501

            # Collate qid and n
            out_qid[batch_index] = int(sample["qid"])
            out_n[batch_index] = min(int(sample["n"]), list_size)

        if sparse:
            ind = _torch.cat([d[0] for d in out_features], dim=1)
            val = _torch.cat([d[1] for d in out_features], dim=0)
            size = (len(batch), list_size, batch[0]['features'].shape[1])
            out_features = _torch.sparse.FloatTensor(
                ind, val, _torch.Size(size))

        return {
            "features": out_features,
            "relevance": out_relevance,
            "qid": out_qid,
            "n": out_n
        }

    return _collate_fn
