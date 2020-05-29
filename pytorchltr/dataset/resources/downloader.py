from pytorchltr.dataset.resources.util import ChecksumError
from pytorchltr.dataset.resources.util import validate_file
from urllib.request import urlopen
from collections import deque
import os
import hashlib
import logging
import time
import sys


class Downloader:
    def __init__(self, url, target, sha256_checksum=None, force_download=False,
                 chunk_size=32*1024, progress_fn=None, create_dirs=True,
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
            progress_fn: (Optional) a callable progress function.
            create_dirs: (Optional) whether to create directories.
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
        Downloads to given destination. If the file already exists and its
        (optional) sha256 checksum matches, this will skip the download.

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
        except (FileNotFoundError, ChecksumError) as error:
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


def _to_human_readable(b):
    """
    Returns a human-readable string representation of given bytes.

    Arugments:
        b: The bytes as an integer or float
    """
    # Convert to human readble byte format
    byte_unit = deque(["B", "KB", "MB", "GB", "TB"])
    while len(byte_unit) > 1 and b > 1024.0:
        byte_unit.popleft()
        b /= 1024.0

    byte_unit = byte_unit.popleft()
    if b < 10.0 and byte_unit != "B":
        return "%.1f%s" % (b, byte_unit)
    else:
        return "%d%s" % (b, byte_unit)


# Set default progress hook depending on whether the stdout is a terminal.
if sys.stdout.isatty():
    DefaultProgress = TerminalProgress
else:
    DefaultProgress = LoggingProgress
