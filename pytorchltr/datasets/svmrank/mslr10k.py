import os
from typing import Optional

from pytorchltr.utils.downloader import DefaultDownloadProgress
from pytorchltr.utils.downloader import Downloader
from pytorchltr.utils.file import validate_and_download
from pytorchltr.utils.file import extract_zip
from pytorchltr.utils.file import dataset_dir
from pytorchltr.datasets.svmrank.svmrank import SVMRankDataset


class MSLR10K(SVMRankDataset):
    """
    Utility class for downloading and using the MSLR-WEB10K dataset:
    https://www.microsoft.com/en-us/research/project/mslr/.

    This dataset is a smaller sampled version of the MSLR-WEB30K dataset.
    """

    downloader = Downloader(
        url="https://api.onedrive.com/v1.0/shares/s!AtsMfWUz5l8nbOIoJ6Ks0bEMp78/root/content",  # noqa: E501
        target="MSLR-WEB10K.zip",
        sha256_checksum="2902142ea33f18c59414f654212de5063033b707d5c3939556124b1120d3a0ba",  # noqa: E501
        progress_fn=DefaultDownloadProgress(),
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

    splits = {
        "train": "train.txt",
        "test": "test.txt",
        "vali": "vali.txt"
    }

    def __init__(self, location: str = dataset_dir("MSLR10K"),
                 split: str = "train", fold: int = 1, normalize: bool = True,
                 filter_queries: Optional[bool] = None, download: bool = True,
                 validate_checksums: bool = True):
        """
        Args:
            location: Directory where the dataset is located.
            split: The data split to load ("train", "test" or "vali")
            fold: Which data fold to load (1...5)
            normalize: Whether to perform query-level feature
                normalization.
            filter_queries: Whether to filter out queries that
                have no relevant items. If not given this will filter queries
                for the test set but not the train set.
            download: Whether to download the dataset if it does not
                exist.
            validate_checksums: Whether to validate the dataset files
                via sha256.
        """
        # Check if specified split and fold exists.
        if split not in MSLR10K.splits.keys():
            raise ValueError("unrecognized data split '%s'" % str(split))

        if fold not in MSLR10K.per_fold_expected_files.keys():
            raise ValueError("unrecognized data fold '%s'" % str(fold))

        # Validate dataset exists and is correct, or download it.
        validate_and_download(
            location=location,
            expected_files=MSLR10K.per_fold_expected_files[fold],
            downloader=MSLR10K.downloader if download else None,
            validate_checksums=validate_checksums)

        # Only filter queries on non-train splits.
        if filter_queries is None:
            filter_queries = False if split == "train" else True

        # Initialize the dataset.
        datafile = os.path.join(location, "Fold%d" % fold,
                                MSLR10K.splits[split])
        super().__init__(file=datafile, sparse=False, normalize=normalize,
                         filter_queries=filter_queries, zero_based="auto")
