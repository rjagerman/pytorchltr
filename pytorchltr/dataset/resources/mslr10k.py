import os

from pytorchltr.dataset.resources.downloader import DefaultProgress
from pytorchltr.dataset.resources.downloader import Downloader
from pytorchltr.dataset.resources.resource import DownloadableResource
from pytorchltr.dataset.resources.util import extract_zip
from pytorchltr.dataset.svmrank import create_svmranking_collate_fn
from pytorchltr.dataset.svmrank import svmranking_dataset


class MSLR10K(DownloadableResource):
    """
    Utility class for downloading and using the MSLR-WEB10K	dataset:
    https://www.microsoft.com/en-us/research/project/mslr/.

    This dataset is a smaller sampled version of the MSLR-WEB30K dataset.

    Attributes:
        normalize (bool): Whether to perform query-level feature normalization.
    """

    default_downloader = Downloader(
        url="https://api.onedrive.com/v1.0/shares/s!AtsMfWUz5l8nbOIoJ6Ks0bEMp78/root/content",  # noqa: E501
        target="MSLR-WEB10K.zip",
        sha256_checksum="2902142ea33f18c59414f654212de5063033b707d5c3939556124b1120d3a0ba",  # noqa: E501
        progress_fn=DefaultProgress(),
        postprocess_fn=extract_zip)

    per_fold_expected_files = {
        1: [
            {"path": "Fold1/train.txt", "sha256": "6eb3fae4e1186e1242a6520f53a98abdbcde5b926dd19a28e51239284b1d55dc"},  # noqa: E501
            {"path": "Fold1/test.txt", "sha256": "33fe002374a4fce58c4e12863e4eee74745d5672a26f3e4ddacc20ccfe7d6ba0"},  # noqa: E501
            {"path": "Fold1/vali.txt", "sha256": "e86fb3fe7e8a5f16479da7ce04f783ae85735f17f66016786c3ffc797dd9d4db"}  # noqa: E501
        ],
        2: [
            {"path": "Fold2/train.txt", "sha256": "40e4a2fcc237d9c164cbb6a3f2fa91fe6cf7d46a419d2f73e21cf090285659eb"},  # noqa: E501
            {"path": "Fold2/test.txt", "sha256": "44add582ccd674cf63af24d3bf6e1074e87a678db77f00b44c37980a3010917a"},  # noqa: E501
            {"path": "Fold2/vali.txt", "sha256": "33fe002374a4fce58c4e12863e4eee74745d5672a26f3e4ddacc20ccfe7d6ba0"}  # noqa: E501
        ],
        3: [
            {"path": "Fold3/train.txt", "sha256": "f13005ceb8de0db76c93b02ee4b2bded6f925097d3ab7938931e8d07aa72acd7"},  # noqa: E501
            {"path": "Fold3/test.txt", "sha256": "c0a5a3c6bd7790d0b4ff3d5e961d0c8c5f8ff149089ce492540fa63035801b7a"},  # noqa: E501
            {"path": "Fold3/vali.txt", "sha256": "44add582ccd674cf63af24d3bf6e1074e87a678db77f00b44c37980a3010917a"}  # noqa: E501
        ],
        4: [
            {"path": "Fold4/train.txt", "sha256": "6c1677cf9b2ed491e26ac6b8c8ca7dfae9c1a375e2bce8cba6df36ab67ce5836"},  # noqa: E501
            {"path": "Fold4/test.txt", "sha256": "dc6083c24a5f0c03df3c91ad3eed7542694115b998acf046e51432cb7a22b848"},  # noqa: E501
            {"path": "Fold4/vali.txt", "sha256": "c0a5a3c6bd7790d0b4ff3d5e961d0c8c5f8ff149089ce492540fa63035801b7a"}  # noqa: E501
        ],
        5: [
            {"path": "Fold5/train.txt", "sha256": "4249797a2f0f46bff279973f0fb055d4a78f67f337769eabd56e82332c044794"},  # noqa: E501
            {"path": "Fold5/test.txt", "sha256": "e86fb3fe7e8a5f16479da7ce04f783ae85735f17f66016786c3ffc797dd9d4db"},  # noqa: E501
            {"path": "Fold5/vali.txt", "sha256": "dc6083c24a5f0c03df3c91ad3eed7542694115b998acf046e51432cb7a22b848"}  # noqa: E501
        ]
    }

    def __init__(self, location, fold=1, normalize=True,
                 downloader=default_downloader, validate_checksums=True):
        """
        Args:
            location: Directory where the dataset is located.
            fold: The fold to use (1...5) in the MSLR dataset.
            normalize: Whether to perform query-level feature normalization.
            downloader: A downloader for downloading the dataset.
            validate_checksums: Whether to validate the dataset files via
                sha256.
        """
        super().__init__("MSLR-WEB10K", location,
                         expected_files=MSLR10K.per_fold_expected_files[fold],
                         downloader=downloader,
                         validate_checksums=validate_checksums)
        self.normalize = normalize
        self._fold = fold

    def fold_folder(self):
        return "Fold%d" % self._fold

    def collate_fn(self):
        return create_svmranking_collate_fn()

    def train(self):
        return svmranking_dataset(
            os.path.join(self.location, self.fold_folder(), "train.txt"),
            sparse=False, normalize=self.normalize, filter_queries=False)

    def test(self):
        return svmranking_dataset(
            os.path.join(self.location, self.fold_folder(), "test.txt"),
            sparse=False, normalize=self.normalize, filter_queries=True)

    def vali(self):
        return svmranking_dataset(
            os.path.join(self.location, self.fold_folder(), "vali.txt"),
            sparse=False, normalize=self.normalize, filter_queries=True)
