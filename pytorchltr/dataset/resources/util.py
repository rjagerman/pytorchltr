import hashlib
import logging
import os
import tarfile
import zipfile


class ChecksumError(Exception):
    """A ChecksumError occurs when a checksum validation fails.

    Attributes:
        msg (str): Human readable string describing the exception
        expected (str): The expected checksum.
        actual (str): The checksum that occurre
    """
    def __init__(self, expected, actual):
        """
        Args:
            expected (str): The expected checksum.
            actual (str): The actual checksum that occurred.
        """
        self.expected = expected
        self.actual = actual
        self.msg = "expected '%s', but got '%s'" % (expected, actual)


def sha256_checksum(path, chunk_size=4*1024*1024):
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


def validate_file(path, sha256=None):
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
    if not (os.path.exists(path) and os.path.isfile(path)):
        raise FileNotFoundError("could not find expected file '%s'" % path)

    if sha256 is not None:
        actual = sha256_checksum(path)
        if actual != sha256:
            raise ChecksumError(sha256, actual)


def extract_tar(path, destination):
    """
    Extracts the .tar[.gz] file at given path to given destination.

    Args:
        path: The location of the .tar[.gz] file.
        destination: The destination to extract to.
    """
    logging.info("extracting tar file at %s to %s", path, destination)
    with tarfile.open(path) as f:
        f.extractall(destination)


def extract_zip(path, destination):
    """
    Extracts the .zip file at given path to given destination.

    Args:
        path: The location of the .zip file.
        destination: The destination to extract to.
    """
    logging.info("extracting zip file at %s to %s", path, destination)
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(destination)
