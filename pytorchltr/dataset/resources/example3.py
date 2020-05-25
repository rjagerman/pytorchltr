from pytorchltr.dataset.resources.resource import Resource
from pytorchltr.dataset.svmrank import svmranking_dataset
from os import path
import logging


class Example3(Resource):
    def __init__(self, location, download=False, normalize=True,
                 url="http://download.joachims.org/svm_light/examples/example3.tar.gz"):
        """
        Utility class for loading and using the Example3 dataset. This dataset
        is a very small toy sample which is useful as a sanity check for
        testing your code.

        Before using this dataset you must review the licensing agreement on
        http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html.

        Arguments:
            location: Directory where the dataset is located.
            download: Whether to download the dataset automatically.
            normalize: Whether to perform query-level feature normalization.
            url: Where to download the dataset from if download is True.
        """
        super().__init__("example3", location, download, url=url)
        self.normalize = normalize

    def train(self):
        return self.load_svmrank_dataset(
            path.join(self.location, "example3", "train.dat"), cache_name="train",
            normalize=self.normalize, filter_queries=False, sparse=False)

    def test(self):
        return self.load_svmrank_dataset(
            path.join(self.location, "example3", "test.dat"), cache_name="test",
            normalize=self.normalize, filter_queries=True, sparse=False)
