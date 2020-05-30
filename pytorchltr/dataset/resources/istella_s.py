import os

from pytorchltr.dataset.resources.downloader import DefaultProgress
from pytorchltr.dataset.resources.downloader import Downloader
from pytorchltr.dataset.resources.resource import DownloadableResource
from pytorchltr.dataset.resources.util import extract_tar
from pytorchltr.dataset.svmrank import create_svmranking_collate_fn
from pytorchltr.dataset.svmrank import svmranking_dataset


class IstellaS(DownloadableResource):
    """
    Utility class for downloading and using the istella-s dataset:
    http://quickrank.isti.cnr.it/istella-dataset/.

    This dataset is a smaller sampled version of the Istella dataset.

    Attributes:
        normalize (bool): Whether to perform query-level feature normalization.
    """

    default_downloader = Downloader(
        url="http://library.istella.it/dataset/istella-s-letor.tar.gz",
        target="istella-s-letor.tar.gz",
        sha256_checksum="41b21116a3650cc043dbe16f02ee39f4467f9405b37fdbcc9a6a05e230a38981",  # noqa: E501
        progress_fn=DefaultProgress(),
        postprocess_fn=extract_tar)

    default_expected_files = [
        {"path": "sample/train.txt", "sha256": "5cda4187b88b597ca6fa98d6cd3a6d7551ba69ffff700eae63a59fbc8af385b8"},  # noqa: E501
        {"path": "sample/test.txt", "sha256": "2cd1a4f46fa21ea2489073979b5e5913146d1e451f1d6e268bc4e472d39da5d7"},  # noqa: E501
        {"path": "sample/vali.txt", "sha256": "f225dd95772fa65dc685351ff2643945fbd9a9c0e874aa1c538d485595e7c890"}  # noqa: E501
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
        super().__init__("istella-s", location,
                         expected_files=IstellaS.default_expected_files,
                         downloader=downloader,
                         validate_checksums=validate_checksums)
        self.normalize = normalize

    def collate_fn(self):
        return create_svmranking_collate_fn()

    def train(self):
        return svmranking_dataset(
            os.path.join(self.location, "sample", "train.txt"), sparse=False,
            normalize=self.normalize, filter_queries=False)

    def test(self):
        return svmranking_dataset(
            os.path.join(self.location, "sample", "test.txt"), sparse=False,
            normalize=self.normalize, filter_queries=True)

    def vali(self):
        return svmranking_dataset(
            os.path.join(self.location, "sample", "vali.txt"), sparse=False,
            normalize=self.normalize, filter_queries=True)
