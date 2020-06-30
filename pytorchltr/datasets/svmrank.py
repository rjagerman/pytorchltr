"""Data loading for SVMRank-style data sets."""
import numpy as _np
import torch as _torch
import logging

from scipy.sparse import coo_matrix as _coo_matrix
from sklearn.datasets import load_svmlight_file as _load_svmlight_file
from torch.utils.data import Dataset as _Dataset


class SVMRankingDataset(_Dataset):
    def __init__(self, file, sparse=False, normalize=False,
                 filter_queries=False, zero_based="auto"):
        """Creates an SVMRank-style dataset from a file.

        Args:
            file (str or file-like): The path to load the dataset from.
            sparse (bool, optional): Whether to load the features as sparse
                features.
            normalize (bool, optional): Whether to perform query-level
                normalization (requires non-sparse features).
            filter_queries (bool, optional): Whether to filter queries that
                have no relevant documents associated with them.
            zero_based (str or int, optional): The zero based index.
        """
        logging.info("loading svmrank dataset from %s", file)

        # Load svmlight file
        self._xs, self._ys, qids = _load_svmlight_file(
            file, query_id=True, zero_based=zero_based)

        # Compute query offsets and unique qids
        self._offsets = _np.hstack(
            [[0], _np.where(qids[1:] != qids[:-1])[0] + 1, [len(qids)]])
        self._unique_qids = qids[self._offsets[:-1]]

        # Densify
        self._sparse = sparse
        if not sparse:
            self._xs = self._xs.A

        # Normalize xs
        if normalize:
            if sparse:
                raise NotImplementedError(
                    "Normalization without dense features is not supported.")
            self._normalize()

        # Filter queries without any relevant documents
        if filter_queries:
            indices = []
            for i, (start, end) in enumerate(zip(self._offsets[:-1],
                                                 self._offsets[1:])):
                if _np.sum(self._ys[start:end]) > 0.0:
                    indices.append(i)
            self._indices = _np.array(indices)
        else:
            self._indices = _np.arange(len(self._unique_qids))

        # Compute qid map and dataset length
        self._qid_map = {
            self._unique_qids[self._indices[index]]: index
            for index in range(len(self._indices))
        }
        self._n = len(self._indices)

    def _normalize(self):
        """Performs query-level feature normalization on the dataset."""
        for start, end in zip(self._offsets[:-1], self._offsets[1:]):
            self._xs[start:end, :] -= _np.min(self._xs[start:end, :], axis=0)
            m = _np.max(self._xs[start:end, :], axis=0)
            m[m == 0.0] = 1.0
            self._xs[start:end, :] /= m

    def get_index(self, qid):
        """Returns the dataset item index for given qid (if it exists).

        Args:
            qid (int): The qid to look up.
        """
        return self._qid_map[qid]

    @staticmethod
    def collate_fn(max_list_size=None, rng=None):
        """Returns a collate_fn that can be used to collate batches.

        Args:
            max_list_size (int, optional): If set, list size per query will be
                truncated to this size.
            rng (:obj:`numpy.random.RandomState`, optional): The random number
                generator used to sample documents when truncating query list
                sizes.
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
                    rel = sample["relevance"][rng_indices]
                    rel_n = len(rng_indices)
                    out_relevance[batch_index, 0:rel_n] = rel
                else:
                    rel = sample["relevance"]
                    rel_n = len(sample["relevance"])
                    out_relevance[batch_index, 0:rel_n] = rel

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

    def __getitem__(self, index):
        r"""
        Returns the item at given index.

        Args:
            index (int): The index.

        Returns:
            bool:
                A dictionary with the LTR item at given index. The structure of
                the dict is as follows:

                .. code-block::

                    {
                        "features": tensor of shape (list_size),
                        "relevance": tensor of shape (list_size),
                        "qid": int indicating the query identifier,
                        "n": int indicating the number of documents,
                        "sparse": true if the tensors are sparse,
                    }

        """
        # Extract query features and relevance labels
        qid = self._unique_qids[self._indices[index]]
        start = self._offsets[self._indices[index]]
        end = self._offsets[self._indices[index] + 1]
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
        r"""
        Returns:
            int:
                The length of the dataset.
        """
        return self._n
