from typing import Optional
import torch as _torch
import numpy as _np


class ListSampler:
    def __init__(self, max_list_size: Optional[int] = None):
        self._max_list_size = max_list_size

    def max_list_size(self, relevance):
        size = relevance.shape[0]
        if self._max_list_size is not None:
            size = min(self._max_list_size, size)
        return size

    def __call__(self, relevance: _torch.LongTensor) -> _torch.LongTensor:
        return _torch.arange(self.max_list_size(relevance), dtype=_torch.long)


class UniformSampler(ListSampler):
    def __init__(self, max_list_size: Optional[int] = None,
                 generator: Optional[_torch.Generator] = None):
        super().__init__(max_list_size)
        self.rng_kw = {"generator": generator} if generator is not None else {}

    def __call__(self, relevance: _torch.LongTensor) -> _torch.LongTensor:
        perm = _torch.randperm(relevance.shape[0], **self.rng_kw)
        return perm[0:self.max_list_size(relevance)]


class BalancedRelevanceSampler(UniformSampler):
    def __init__(self, max_list_size: Optional[int] = None,
                 rng: Optional[_np.random.RandomState] = None):
        super().__init__(max_list_size, rng)

    def __call__(self, relevance: _torch.LongTensor) -> _torch.LongTensor:
        unique_rel = _torch.unique(relevance)
        max_size = self.max_list_size(relevance)
        stacked_rel = -_torch.ones((unique_rel.shape[0], relevance.shape[0]),
                                   dtype=_torch.long, device=relevance.device)
        perm = _torch.randperm(relevance.shape[0], **self.rng_kw)
        rel_shuffled = relevance[perm]
        unique_perm = _torch.randperm(unique_rel.shape[0], **self.rng_kw)
        for j, rel in enumerate(unique_rel[unique_perm]):
            idxs = _torch.where(rel_shuffled == rel)[0]
            size = min(max_size, idxs.shape[0])
            stacked_rel[j, 0:size] = idxs[0:size]
        out = stacked_rel.T.reshape(relevance.shape[0] * unique_rel.shape[0])
        out = out[out != -1]
        out = perm[out]
        return out[0:max_size]
