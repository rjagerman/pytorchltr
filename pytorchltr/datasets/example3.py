import os
from typing import Optional

from pytorchltr.utils.downloader import DefaultDownloadProgress
from pytorchltr.utils.downloader import Downloader
from pytorchltr.utils.file import validate_and_download
from pytorchltr.utils.file import extract_tar
from pytorchltr.utils.file import dataset_dir
from pytorchltr.datasets.svmrank import SVMRankDataset


class Example3(SVMRankDataset):
    """
    Utility class for loading and using the Example3 dataset:
    http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html

    This dataset is a very small toy sample which is useful as a sanity check
    for testing your code.
    """

    downloader = Downloader(
        url="http://download.joachims.org/svm_light/examples/example3.tar.gz",
        target="example3.tar.gz",
        sha256_checksum="c46e97b66d3c9d5f37f7c3a2201aa2c4ea2a4e8a768f8794b10152c22648106b",  # noqa: E501
        progress_fn=DefaultDownloadProgress(),
        postprocess_fn=extract_tar)

    expected_files = [
        {"path": os.path.join("example3", "train.dat"), "sha256": "503aa66c6a1b1bb8a86b14e52163dcdb5bcffc017981afdff4cf026eacc592cf"},  # noqa: E501
        {"path": os.path.join("example3", "test.dat"), "sha256": "81aaac13dfc5180edce38a588cec80ee00b5d85662e00d1b7ac1d3f98242698e"}  # noqa: E501
    ]

    splits = {
        "train": os.path.join("example3", "train.dat"),
        "test": os.path.join("example3", "test.dat")
    }

    def __init__(self, location: str = dataset_dir("example3"),
                 split: str = "train",
                 normalize: bool = True, filter_queries: Optional[bool] = None,
                 download: bool = True, validate_checksums: bool = True):
        """
        Args:
            location: Directory where the dataset is located.
            split: The data split to load ("train" or "test")
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
        if split not in Example3.splits.keys():
            raise ValueError("unrecognized data split '%s'" % split)

        # Validate dataset exists and is correct, or download it.
        validate_and_download(
            location=location,
            expected_files=Example3.expected_files,
            downloader=Example3.downloader if download else None,
            validate_checksums=validate_checksums)

        # Only filter queries on non-train splits.
        if filter_queries is None:
            filter_queries = False if split == "train" else True

        # Initialize the dataset.
        super().__init__(file=os.path.join(location, Example3.splits[split]),
                         sparse=False, normalize=normalize,
                         filter_queries=filter_queries, zero_based="auto")
