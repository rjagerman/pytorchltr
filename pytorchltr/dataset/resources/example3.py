import os

from pytorchltr.dataset.resources.downloader import DefaultProgress
from pytorchltr.dataset.resources.downloader import Downloader
from pytorchltr.dataset.resources.resource import DownloadableResource
from pytorchltr.dataset.resources.util import extract_tar
from pytorchltr.dataset.svmrank import create_svmranking_collate_fn
from pytorchltr.dataset.svmrank import svmranking_dataset


class Example3(DownloadableResource):
    """
    Utility class for loading and using the Example3 dataset:
    http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html

    This dataset is a very small toy sample which is useful as a sanity check
    for testing your code.
    """

    default_downloader = Downloader(
        url="http://download.joachims.org/svm_light/examples/example3.tar.gz",
        target="example3.tar.gz",
        sha256_checksum="c46e97b66d3c9d5f37f7c3a2201aa2c4ea2a4e8a768f8794b10152c22648106b",
        progress_fn=DefaultProgress(),
        postprocess_fn=extract_tar)

    default_expected_files = [
        {"path": "example3/train.dat", "sha256": "503aa66c6a1b1bb8a86b14e52163dcdb5bcffc017981afdff4cf026eacc592cf"},
        {"path": "example3/test.dat", "sha256": "81aaac13dfc5180edce38a588cec80ee00b5d85662e00d1b7ac1d3f98242698e"}
    ]

    def __init__(self, location, normalize=True, downloader=default_downloader,
                 validate_checksums=True):
        """
        Args:
            location: Directory where the dataset is located.
            normalize: Whether to perform query-level feature normalization.
            downloader: A downloader for downloading the dataset.
            validate_checksums: Whether to validate the dataset files via
                sha256.
        """
        super().__init__("example3", location,
                         expected_files=Example3.default_expected_files,
                         downloader=downloader,
                         validate_checksums=validate_checksums)
        self.normalize = normalize

    def collate_fn(self):
        return create_svmranking_collate_fn()

    def train(self):
        return svmranking_dataset(
            os.path.join(self.location, "example3", "train.dat"), sparse=False,
            normalize=self.normalize, filter_queries=False)

    def test(self):
        return svmranking_dataset(
            os.path.join(self.location, "example3", "test.dat"), sparse=False,
            normalize=self.normalize, filter_queries=True)
