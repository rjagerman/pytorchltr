import os
from typing import Optional

from pytorchltr.utils.downloader import DefaultDownloadProgress
from pytorchltr.utils.downloader import Downloader
from pytorchltr.utils.file import validate_and_download
from pytorchltr.utils.file import extract_zip
from pytorchltr.utils.file import dataset_dir
from pytorchltr.datasets.svmrank import SVMRankDataset


class MSLR30K(SVMRankDataset):
    """
    Utility class for downloading and using the MSLR-WEB30K dataset:
    https://www.microsoft.com/en-us/research/project/mslr/.
    """

    downloader = Downloader(
        url="https://api.onedrive.com/v1.0/shares/s!AtsMfWUz5l8nbXGPBlwD1rnFdBY/root/content",  # noqa: E501
        target="MSLR-WEB30K.zip",
        sha256_checksum="08cb7977e1d5cbdeb57a9a2537a0923dbca6d46a76db9a6afc69e043c85341ae",  # noqa: E501
        progress_fn=DefaultDownloadProgress(),
        postprocess_fn=extract_zip)

    per_fold_expected_files = {
        1: [
            {"path": "Fold1/train.txt", "sha256": "40b8eee4d1221cf8205e81603441c1757dd024a3baac25a47756210b03c031d6"},  # noqa: E501
            {"path": "Fold1/test.txt", "sha256": "9a4668fd2615e6772d2e5c4d558d084b2daaf2405571eaf3e4d0526f4da096c7"},  # noqa: E501
            {"path": "Fold1/vali.txt", "sha256": "7647834b84c849a61e5cf3c999a2f72a4785613286fd972a5615e9fcb58f94d8"}  # noqa: E501
        ],
        2: [
            {"path": "Fold2/train.txt", "sha256": "a6d12dc4cc8c2dd0743b58c49cad12bd0f6f1cbeda54b54a91e5b54e0b96e7ca"},  # noqa: E501
            {"path": "Fold2/test.txt", "sha256": "d192c023baebdae148d902d716aeebf3df2f2f1ce5aee12be4f0b8bb76a8c04a"},  # noqa: E501
            {"path": "Fold2/vali.txt", "sha256": "9a4668fd2615e6772d2e5c4d558d084b2daaf2405571eaf3e4d0526f4da096c7"}  # noqa: E501
        ],
        3: [
            {"path": "Fold3/train.txt", "sha256": "3d0eb52a6702b2c48750a6de89e757cfac499c8ec1d38e3843cb531c059f2f74"},  # noqa: E501
            {"path": "Fold3/test.txt", "sha256": "ba2487a27c21dceea0b9afaadbb6b392d5f652eb0fb8649cbd201bb894c47c12"},  # noqa: E501
            {"path": "Fold3/vali.txt", "sha256": "d192c023baebdae148d902d716aeebf3df2f2f1ce5aee12be4f0b8bb76a8c04a"}  # noqa: E501
        ],
        4: [
            {"path": "Fold4/train.txt", "sha256": "21e389656be3c2bfe92eb4fb898d2f30dc24990fc524aada7471b949f176778d"},  # noqa: E501
            {"path": "Fold4/test.txt", "sha256": "a7ba03d708ae6b21556a8a6859d52e37cc0dcb2cbff992c630ef6173bb02122a"},  # noqa: E501
            {"path": "Fold4/vali.txt", "sha256": "ba2487a27c21dceea0b9afaadbb6b392d5f652eb0fb8649cbd201bb894c47c12"}  # noqa: E501
        ],
        5: [
            {"path": "Fold5/train.txt", "sha256": "a19da1799650b08c0dc3baaec6c590469ef1773148082e1a10b8504f2f5e9a8b"},  # noqa: E501
            {"path": "Fold5/test.txt", "sha256": "7647834b84c849a61e5cf3c999a2f72a4785613286fd972a5615e9fcb58f94d8"},  # noqa: E501
            {"path": "Fold5/vali.txt", "sha256": "a7ba03d708ae6b21556a8a6859d52e37cc0dcb2cbff992c630ef6173bb02122a"}  # noqa: E501
        ]
    }

    splits = {
        "train": "train.txt",
        "test": "test.txt",
        "vali": "vali.txt"
    }

    def __init__(self, location: str = dataset_dir("MSLR30K"),
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
        if split not in MSLR30K.splits.keys():
            raise ValueError("unrecognized data split '%s'" % str(split))

        if fold not in MSLR30K.per_fold_expected_files.keys():
            raise ValueError("unrecognized data fold '%s'" % str(fold))

        # Validate dataset exists and is correct, or download it.
        validate_and_download(
            location=location,
            expected_files=MSLR30K.per_fold_expected_files[fold],
            downloader=MSLR30K.downloader if download else None,
            validate_checksums=validate_checksums)

        # Only filter queries on non-train splits.
        if filter_queries is None:
            filter_queries = False if split == "train" else True

        # Initialize the dataset.
        datafile = os.path.join(location, "Fold%d" % fold,
                                MSLR30K.splits[split])
        super().__init__(file=datafile, sparse=False, normalize=normalize,
                         filter_queries=filter_queries, zero_based="auto")
