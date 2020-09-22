import os
from typing import Optional

from pytorchltr.utils.downloader import DefaultDownloadProgress
from pytorchltr.utils.downloader import Downloader
from pytorchltr.utils.file import validate_and_download
from pytorchltr.utils.file import extract_tar
from pytorchltr.utils.file import dataset_dir
from pytorchltr.datasets.svmrank.svmrank import SVMRankDataset


class Istella(SVMRankDataset):
    """
    Utility class for downloading and using the istella dataset:
    http://quickrank.isti.cnr.it/istella-dataset/.
    """

    downloader = Downloader(
        url="http://library.istella.it/dataset/istella-letor.tar.gz",
        target="istella-letor.tar.gz",
        sha256_checksum="d45899d9a6a0e48afb250aac7ee3dc50e73e263687f15761d754515cd8284e0b",  # noqa: E501
        progress_fn=DefaultDownloadProgress(),
        postprocess_fn=extract_tar)

    expected_files = [
        {"path": "full/train.txt", "sha256": "8b08ca4c36281e408bfa026303461a6162398a1eec22fb79db0709c76bf6d189"},  # noqa: E501
        {"path": "full/test.txt", "sha256": "4b34523af8e030718f0216bcae0a35caee1e34e5198ff3e76c08a75db61aac65"},  # noqa: E501
    ]

    splits = {
        "train": "train.txt",
        "test": "test.txt"
    }

    def __init__(self, location: str = dataset_dir("istella"),
                 split: str = "train", normalize: bool = True,
                 filter_queries: Optional[bool] = None, download: bool = True,
                 validate_checksums: bool = True):
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
        if split not in Istella.splits.keys():
            raise ValueError("unrecognized data split '%s'" % str(split))

        # Validate dataset exists and is correct, or download it.
        validate_and_download(
            location=location,
            expected_files=Istella.expected_files,
            downloader=Istella.downloader if download else None,
            validate_checksums=validate_checksums)

        # Only filter queries on non-train splits.
        if filter_queries is None:
            filter_queries = False if split == "train" else True

        # Initialize the dataset.
        datafile = os.path.join(location, "full", Istella.splits[split])
        super().__init__(file=datafile, sparse=False, normalize=normalize,
                         filter_queries=filter_queries, zero_based="auto")
