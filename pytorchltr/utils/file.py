import hashlib
import logging
import os
import tarfile
import zipfile
from typing import Dict
from typing import List
from typing import Optional
from typing import Union


class ChecksumError(Exception):
    """A ChecksumError occurs when a checksum validation fails.

    Attributes:
        msg (str): Human readable string describing the exception
        origin (str): The origin of the checksum error.
        expected (str): The expected checksum.
        actual (str): The checksum that occurre
    """
    def __init__(self, origin: str, expected: str, actual: str):
        """
        Args:
            origin: The origin of the checksum error.
            expected: The expected checksum.
            actual: The actual checksum that occurred.
        """
        self.origin = origin
        self.expected = expected
        self.actual = actual
        self.msg = "'%s' checksum error: expected '%s', but got '%s'" % (
            origin, expected, actual)


def sha256_checksum(path: str, chunk_size: int = 4*1024*1024):
    """
    Computes the sha256 checksum on the file in given path.

    Args:
        path: The file to compute the sha256 checksum for.
        chunk_size: Chunk size to read per time, prevents loading the full file
            into memory (default 4MiB).

    Returns:
        The sha256 checksum hex digest as a string.
    """
    hash_sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def validate_file(path: str, sha256: Optional[str] = None):
    """
    Performs a validation check on the file in given path:
     * Checks if the file exists.
     * (Optional) Checks if the file's sha256 matches given sha256.

    This function will raise a FileNotFoundError if the given file does not
    exist or it will raise an AssertionError if the sha256 checksum fails. If
    no exception is raised, all checks have passed.

    Args:
        path: The path to the file to check.
        sha256: (Optional) the sha256 checksum to compare against.
    """
    logging.debug("checking if '%s' exists and is a file", path)
    if not (os.path.exists(path) and os.path.isfile(path)):
        raise FileNotFoundError("could not find expected file '%s'" % path)

    if sha256 is not None:
        logging.debug("checking sha256 checksum of '%s'", path)
        actual = sha256_checksum(path)
        if actual != sha256:
            raise ChecksumError(path, sha256, actual)


def validate_expected_files(location: str,
                            expected_files: List[Union[str, Dict[str, str]]],
                            validate_checksums: bool = False):
    """
    Performs a validation check on a set of expected files:
     * Checks if all files exists
     * (Optional) Checks if each file's sha256 matches given sha256.

    This function will raise a FileNotFoundError or a ChecksumError if
    something is missing or if a checksum fails. A RuntimeError may be raised
    if expected_files has an incorrect format.

    Args:
        location: The directory to check
        expected_files: A list of expected files for this resource. Each entry
            in the list should be either a string indicating the path of the
            file or a dict containing a 'path' and 'sha256' key for the path
            and sha256 checksum of the file.
        validate_checksums: Whether to validate checksums or skip them.
    """
    if expected_files is not None:
        for f in expected_files:
            if isinstance(f, str):
                validate_file(os.path.join(location, f))
            elif isinstance(f, dict) and "path" in f and "sha256" in f:
                validate_file(
                    os.path.join(location, f["path"]),
                    f["sha256"] if validate_checksums else None)
            else:
                raise ValueError(
                    "entries in expected_files should be either of type "
                    "str or a dict containing 'path' and 'sha256' keys.")


_DOWNLOADER_TYPE = "pytorchltr.utils.downloader.Downloader"


def validate_and_download(location: str,
                          expected_files: List[Union[str, Dict[str, str]]],
                          downloader: Optional[_DOWNLOADER_TYPE] = None,
                          validate_checksums: bool = False):
    """Validates expected files at given location and attempts to download
    if validation fails.

    Args:
        location: The location to check.
        expected_files: (Optional) a list of expected files for this resource.
            Each entry in the list should be either a string indicating the
            path of the file or a dict containing a 'path' and 'sha256' key for
            the path and sha256 checksum of the file.
        downloader: The downloader to use when downloading files.
        validate_checksums: Whether to validate checksums.
    """
    try:
        logging.info("checking dataset files in '%s'", location)
        validate_expected_files(
            location, expected_files, validate_checksums)
        logging.info("successfully checked all dataset files")
    except (FileNotFoundError, ChecksumError):
        logging.warning("dataset file(s) in '%s' are missing or corrupt",
                        location)
        if downloader is not None:
            downloader.download(location)
            validate_expected_files(
                location, expected_files, validate_checksums)
            logging.info("successfully checked all dataset files")
        else:
            raise


def extract_tar(path: str, destination: str):
    """
    Extracts the .tar[.gz] file at given path to given destination.

    Args:
        path: The location of the .tar[.gz] file.
        destination: The destination to extract to.
    """
    logging.info("extracting tar file at '%s' to '%s'", path, destination)
    with tarfile.open(path) as f:
        f.extractall(destination)


def extract_zip(path: str, destination: str):
    """
    Extracts the .zip file at given path to given destination.

    Args:
        path: The location of the .zip file.
        destination: The destination to extract to.
    """
    logging.info("extracting zip file at '%s' to '%s'", path, destination)
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(destination)


def dataset_dir(name: str) -> str:
    """
    Returns the location of the dataset directory.

    Args:
        name: The name of the dataset.

    Returns:
        The path to the dataset directory.
    """
    dataset_path = os.path.join(os.environ.get("HOME", "."),
                                ".pytorchltr_datasets")
    dataset_path = os.environ.get("DATASET_PATH", dataset_path)
    dataset_path = os.environ.get("PYTORCHLTR_DATASET_PATH", dataset_path)
    return os.path.join(dataset_path, name)
