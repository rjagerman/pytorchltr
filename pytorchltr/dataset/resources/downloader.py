import logging
import os
import sys
import time
from collections import deque
from urllib.request import urlopen

from pytorchltr.dataset.resources.util import ChecksumError
from pytorchltr.dataset.resources.util import validate_file


class Downloader:
    """
    Downloader allows downloading from a given url to a given target file.

    Attributes:
        url (str): The URL to download from.
        target (str): The target file name to save as.
        sha256_checksum (str, optional): The SHA256 checksum of the file which
            will be checked to see if a download and is needed and to validate
            the downloaded contents.
        force_download (bool): If set to True, a call to download will always
            download, regardless of whether the target file already exists.
        chunk_size (int): The download chunk size, which is the maximum amount
            of bytes read per file-write. Default is 32*1024 (32KB)
        create_dirs (bool): Whether to create directories automatically if the
            target file should be downloaded in a non-existing directory.
        progress_fn (callable, optional): A callable function that reports the
            download progress after every download chunk.
        postprocess_fn (callable, optional): A callable function that is called
            when the download finished. Typical use cases include extracting or
            unzipping a downloaded archive.
    """
    def __init__(self, url, target, sha256_checksum=None, force_download=False,
                 chunk_size=32*1024, create_dirs=True, progress_fn=None,
                 postprocess_fn=None):
        """
        Creates a downloader that downloads from a url.

        Unless force_download is set to True, this class will skip the download
        if destination already exists and its (optional) sha256 checksum
        matches.

        Args:
            url: The url to download from.
            target: The target name to save.
            sha256_checksum: (Optional) sha256 checksum to validate against.
            force_download: (Optional) forces a re-dowload always.
            chunk_size: (Optional) chunk size to download in.
            create_dirs: (Optional) whether to create directories.
            progress_fn: (Optional) a callable progress function.
            postprocess_fn: (Optional) function to call after successfull
                download.
        """
        self.url = url
        self.target = target
        self.sha256_checksum = sha256_checksum
        self.force_download = force_download
        self.chunk_size = chunk_size
        self.progress_fn = progress_fn
        self.create_dirs = create_dirs
        self.postprocess_fn = postprocess_fn

    def download(self, destination):
        """
        Downloads to given destination. Unless force_download is True, this
        will skip the download if the file already exists and its (optional)
        sha256 checksum matches.

        Arguments:
            destination: The destination to download to
        """
        path = os.path.join(destination, self.target)
        if self.force_download or self._should_download(path):
            self._download(path)
            validate_file(path, self.sha256_checksum)
        if self.postprocess_fn is not None:
            self.postprocess_fn(path, destination)

    def _should_download(self, path):
        """
        Checks if the file should be downloaded.

        Args:
            path: The path to check.
        """
        try:
            validate_file(path, self.sha256_checksum)
            return False
        except (FileNotFoundError, ChecksumError):
            return True

    def _download(self, path):
        """
        Downloads the file from self.url to path.

        Args:
            path: The path to download the file to.
        """
        # Create directories
        if self.create_dirs:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # Open URL
        with urlopen(self.url) as response:

            # Get total size if response provides it (otherwise None)
            total_size = response.info().get("Content-Length")
            bytes_read = 0
            if total_size:
                total_size = int(total_size.strip())

            # Download to destination
            with open(path, "wb") as f:
                self._progress(bytes_read, total_size, False)
                for chunk in iter(lambda: response.read(self.chunk_size), b""):
                    f.write(chunk)
                    bytes_read += len(chunk)
                    self._progress(bytes_read, total_size, False)
                self._progress(bytes_read, total_size, True)

    def _progress(self, bytes_read, total_size, final):
        """
        Reports download progress to the progress hook if it exists.

        Arguments:
            bytes_read: How many bytes have been read so far.
            total_size: The total size of the download (None if unknown).
            final: True when the download finishes, False otherwise.
        """
        if self.progress_fn is not None:
            self.progress_fn(bytes_read, total_size, final)


class IntervalProgress:
    """
    A progress hook function that reports to output at a specified interval.
    """
    def __init__(self, interval=1.0):
        self.interval = interval
        self.last_update = time.time() - interval

    def __call__(self, bytes_read, total_size, final):
        if final or time.time() - self.last_update >= self.interval:
            self.progress(bytes_read, total_size, final)
            self.last_update = time.time()

    def progress(self, bytes_read, total_size, final):
        """Processes the progress so far. Called only once per interval.

        Args:
            bytes_read (int): The number of bytes read so far.
            total_size (int, optional): The total number of bytes.
            final (bool): Whether this is the final progress call.
        """
        raise NotImplementedError


class LoggingProgress(IntervalProgress):
    """
    An interval progress hook that reports to logging.info.
    """
    def progress(self, bytes_read, total_size, final):
        logging.info(_progress_string(bytes_read, total_size, final))


class TerminalProgress(IntervalProgress):
    """
    An interval progress hook that writes to the terminal via print.
    """
    def progress(self, bytes_read, total_size, final):
        print("\033[K" + _progress_string(bytes_read, total_size, final),
              end="\n" if final else "\r")


def _progress_string(bytes_read, total_size, final):
    """
    Returns a human-readable string representing the download progress.

    Arguments:
        bytes_read: The number of bytes read so far.
        total_size: The total number of bytes to read or None if unknown.
    """
    if final:
        return "finished downloading [%s]" % _to_human_readable(bytes_read)
    if total_size is None:
        return "downloading [%s / ?]" % _to_human_readable(bytes_read)
    else:
        percent = int((100.0 * bytes_read) / total_size)
        return "downloading %3d%% [%s / %s]" % (
            percent,
            _to_human_readable(bytes_read),
            _to_human_readable(total_size))


def _to_human_readable(nr_of_bytes):
    """
    Returns a human-readable string representation of given bytes.

    Arugments:
        nr_of_bytes: The number bytes as an integer or float
    """
    # Convert to human readble byte format
    byte_unit = deque(["B", "KB", "MB", "GB", "TB"])
    while len(byte_unit) > 1 and nr_of_bytes > 1024.0:
        byte_unit.popleft()
        nr_of_bytes /= 1024.0

    byte_unit = byte_unit.popleft()
    if nr_of_bytes < 10.0 and byte_unit != "B":
        return "%.1f%s" % (nr_of_bytes, byte_unit)
    return "%d%s" % (nr_of_bytes, byte_unit)


# Set default progress hook depending on whether the stdout is a terminal.
if sys.stdout.isatty():
    DefaultProgress = TerminalProgress
else:
    DefaultProgress = LoggingProgress
