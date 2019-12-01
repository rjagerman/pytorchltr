from torch.utils.data import Dataset as _Dataset
from sklearn.datasets import load_svmlight_file as _load_svmlight_file
from scipy.sparse import coo_matrix as _coo_matrix
import numpy as _np
import torch as _torch


class SVMRankingDataset(_Dataset):
    def __init__(self, xs, ys, unique_qids, offsets, sparse):
        r"""An SVM ranking dataset supporting both dense and sparse tensors.

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
        self._offsets = offsets
        self._n = len(self._offsets) - 1
        self._sparse = sparse

    def __getitem__(self, index):
        # Extract query features and relevance labels
        qid = self._unique_qids[index]
        start = self._offsets[index]
        end = self._offsets[index + 1]
        features = self._xs[start:end, :]
        y = _torch.tensor(self._ys[start:end])
        n = end - start

        # Compute sparse or dense torch tensor
        if self._sparse:
            coo = _coo_matrix(features)
            ind = _torch.LongTensor(_np.vstack((coo.row, coo.col)))
            val = _torch.FloatTensor(coo.data)
            features = _torch.sparse.FloatTensor(
                ind, val, _torch.Size(coo.shape))
        else:
            features = _torch.tensor(features)

        # Return data sample
        return {
            "features": features,
            "relevance": y,
            "qid": qid,
            "n": n
        }

    def __len__(self):
        return self._n


def svmranking_dataset(file, sparse=False, normalize=False,
                       zero_based="auto"):
    r"""Loads an SVMRank-style dataset from given file_path.

    Arguments:
        file (str or file-like): The path to load the dataset from.
        sparse (bool, optional): Whether to load the features as sparse features.
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
            raise NotImplementedError("Normalization without dense features is not supported.")
        _per_offset_normalize(xs, offsets)

    # Return full dataset
    return SVMRankingDataset(xs, ys, unique_qids, offsets, sparse)


def _per_offset_normalize(xs, offsets):
    for start, end in zip(offsets[:-1], offsets[1:]):
        xs[start:end,:] -= _np.min(xs[start:end,:], axis=0)
        m = _np.max(xs[start:end,:], axis=0)
        m[m == 0.0] = 1.0
        xs[start:end,:] /= m


def collate_svmrank(batch):
    pass
