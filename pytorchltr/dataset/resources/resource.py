import logging
import os

from pytorchltr.dataset.resources.util import ChecksumError
from pytorchltr.dataset.resources.util import validate_file


class Resource:
    """A resource provides access to a publically available dataset.

    A resource provides convenient access to the train, vali and/or test splits
    provided by publically available datasets.
    """
    def __init__(self, name, location, expected_files=None):
        """
        Initializes the resource with given name in directory at root.

        Args:
            name: The name of the resource.
            location: The directory to load the resource from.
            expected_files: (Optional) a list of expected files for this
                resource. Each entry in the list should be either a string
                indicating the path of the file or a dict containing a 'path'
                and 'sha256' key for the path and sha256 checksum of the file.
        """
        self.name = name
        self.location = location
        self.expected_files = expected_files

    def validate_resource(self, validate_checksums=True):
        """
        Validates whether the resource at specified location is correct,
        meaning that all expected files exist and their (optional) sha256
        checksums match.

        Args:
            validate_checksums: If True, this will additionally validate
                file checksums via sha256.
        """
        if self.expected_files is not None:
            for f in self.expected_files:
                if isinstance(f, str):
                    validate_file(os.path.join(self.location, f))
                elif isinstance(f, dict) and "path" in f and "sha256" in f:
                    validate_file(
                        os.path.join(self.location, f["path"]),
                        f["sha256"] if validate_checksums else None)
                else:
                    raise RuntimeError("entries in expected_files should be either of type str or a dict containing 'path' and 'sha256' keys.")

    def collate_fn(self):
        """Returns the collate function used for batching this dataset."""
        raise NotImplementedError("%s provides no batch collate function" % self.name)

    def train(self):
        """Returns the train split of the resource."""
        raise NotImplementedError("%s provides no train split" % self.name)

    def test(self):
        """Returns the test split of the resource."""
        raise NotImplementedError("%s provides no test split" % self.name)

    def vali(self):
        """Returns the validation split of the resource."""
        raise NotImplementedError("%s provides no vali split" % self.name)


class DownloadableResource(Resource):
    """A Downloadable Resource provides immediate download functionality to a
    resource."""
    def __init__(self, name, location, expected_files=None, downloader=None,
                 validate_checksums=True):
        """
        Creates a downloadable resource

        Args:
            name (str): The name of the resource.
            location (str): The directory to load the resource from.
            expected_files: A list of expected files for this resource. Each
                entry in the list should be either a string indicating the path
                of the file or a dict containing a 'path' and 'sha256' key for
                the path and sha256 checksum of the file respectively.
            downloader: (Optional) A downloader object that is used to download
                and unpack the resource.
            validate_checksums: (Optional) Whether to perform checksum
                validation on downloaded files.

        Raises:
            FileNotFoundError: If an expected resource file is missing.
            AssertionError: If a resource file has a checksum mismatch.
        """
        super().__init__(name, location, expected_files=expected_files)
        try:
            self.validate_resource(validate_checksums)
        except (FileNotFoundError, ChecksumError):
            if downloader is not None:
                logging.info("attempting to download %s dataset", self.name)
                downloader.download(location)
            self.validate_resource(validate_checksums)
