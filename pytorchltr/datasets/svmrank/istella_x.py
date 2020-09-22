import os
from typing import Optional

from pytorchltr.utils.downloader import DefaultDownloadProgress
from pytorchltr.utils.downloader import Downloader
from pytorchltr.utils.file import validate_and_download
from pytorchltr.utils.file import extract_tar
from pytorchltr.utils.file import dataset_dir
from pytorchltr.datasets.svmrank.svmrank import SVMRankDataset


class IstellaX(SVMRankDataset):
    """
    Utility class for downloading and using the istella-X dataset:
    http://quickrank.isti.cnr.it/istella-dataset/.
    """

    downloader = Downloader(
        url="http://quickrank.isti.cnr.it/istella-datasets-mirror/istella-X.tar.gz",  # noqa: E501
        target="istella-X.tar.gz",
        sha256_checksum="e67be60d1c6a68983a669e4de9df2c395914717b2017cb3b68a97eb89f2ea763",  # noqa: E501
        progress_fn=DefaultDownloadProgress(),
        postprocess_fn=extract_tar)

    expected_files = [
        {"path": "train.txt", "sha256": "710378b3536a2156ae0747f0683e2e811e992e7a12014408ff93fbb1ea5a3340"},  # noqa: E501
        {"path": "test.txt", "sha256": "c2b808bae0ccbc40df9519b6a3906a297c8207eb273304c22d506a8612fa9eef"},  # noqa: E501
        {"path": "vali.txt", "sha256": "9de516e2e0dfd0e0a29cf4b233df8e22f2b547f260f7881c7dc1c7237101ee7e"}  # noqa: E501
    ]

    splits = {
        "train": "train.txt",
        "test": "test.txt",
        "vali": "vali.txt"
    }

    def __init__(self, location: str = dataset_dir("istella_x"),
                 split: str = "train", normalize: bool = True,
                 filter_queries: Optional[bool] = None, download: bool = True,
                 validate_checksums: bool = True):
        """
        Args:
            location: Directory where the dataset is located.
            split: The data split to load ("train", "test" or "vali")
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
        # Check if specified split exists.
        if split not in IstellaX.splits.keys():
            raise ValueError("unrecognized data split '%s'" % str(split))

        # Validate dataset exists and is correct, or download it.
        validate_and_download(
            location=location,
            expected_files=IstellaX.expected_files,
            downloader=IstellaX.downloader if download else None,
            validate_checksums=validate_checksums)

        # Only filter queries on non-train splits.
        if filter_queries is None:
            filter_queries = False if split == "train" else True

        # Initialize the dataset.
        datafile = os.path.join(location, IstellaX.splits[split])
        super().__init__(file=datafile, sparse=False, normalize=normalize,
                         filter_queries=filter_queries, zero_based="auto")
