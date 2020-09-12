import os
from typing import Optional

from pytorchltr.utils.downloader import DefaultDownloadProgress
from pytorchltr.utils.downloader import Downloader
from pytorchltr.utils.file import validate_and_download
from pytorchltr.utils.file import extract_tar
from pytorchltr.utils.file import dataset_dir
from pytorchltr.datasets.svmrank.svmrank import SVMRankDataset


class IstellaS(SVMRankDataset):
    """
    Utility class for downloading and using the istella-s dataset:
    http://quickrank.isti.cnr.it/istella-dataset/.

    This dataset is a smaller sampled version of the Istella dataset.
    """

    downloader = Downloader(
        url="http://library.istella.it/dataset/istella-s-letor.tar.gz",
        target="istella-s-letor.tar.gz",
        sha256_checksum="41b21116a3650cc043dbe16f02ee39f4467f9405b37fdbcc9a6a05e230a38981",  # noqa: E501
        progress_fn=DefaultDownloadProgress(),
        postprocess_fn=extract_tar)

    expected_files = [
        {"path": "sample/train.txt", "sha256": "5cda4187b88b597ca6fa98d6cd3a6d7551ba69ffff700eae63a59fbc8af385b8"},  # noqa: E501
        {"path": "sample/test.txt", "sha256": "2cd1a4f46fa21ea2489073979b5e5913146d1e451f1d6e268bc4e472d39da5d7"},  # noqa: E501
        {"path": "sample/vali.txt", "sha256": "f225dd95772fa65dc685351ff2643945fbd9a9c0e874aa1c538d485595e7c890"}  # noqa: E501
    ]

    splits = {
        "train": "train.txt",
        "test": "test.txt",
        "vali": "vali.txt"
    }

    def __init__(self, location: str = dataset_dir("istella_s"),
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
        if split not in IstellaS.splits.keys():
            raise ValueError("unrecognized data split '%s'" % str(split))

        # Validate dataset exists and is correct, or download it.
        validate_and_download(
            location=location,
            expected_files=IstellaS.expected_files,
            downloader=IstellaS.downloader if download else None,
            validate_checksums=validate_checksums)

        # Only filter queries on non-train splits.
        if filter_queries is None:
            filter_queries = False if split == "train" else True

        # Initialize the dataset.
        datafile = os.path.join(location, "sample", IstellaS.splits[split])
        super().__init__(file=datafile, sparse=False, normalize=normalize,
                         filter_queries=filter_queries, zero_based="auto")
