from pytorchltr.dataset.resources.resource import Resource
from pytorchltr.dataset.svmrank import svmranking_dataset
from os import path
import logging


class IstellaS(Resource):
    def __init__(self, location, download=False, normalize=True,
                 url="http://library.istella.it/dataset/istella-s-letor.tar.gz"):
        """
        Utility class for loading and using the istella-s dataset.

        Before using this dataset you must review the licensing agreement on
        http://quickrank.isti.cnr.it/istella-dataset/.

        Arguments:
            location: Directory where the dataset is located.
            download: Whether to download the dataset automatically.
            normalize: Whether to perform query-level feature normalization.
            url: Where to download the dataset from if download is True.
        """
        super().__init__("istella-s", location, download, url=url)
        self.normalize = normalize

    def train(self):
        return self.load_svmrank_dataset(
            path.join(self.location, "sample", "train.txt"), cache_name="train",
            normalize=self.normalize, filter_queries=False, sparse=False)

    def test(self):
        return self.load_svmrank_dataset(
            path.join(self.location, "sample", "test.txt"), cache_name="test",
            normalize=self.normalize, filter_queries=True, sparse=False)
