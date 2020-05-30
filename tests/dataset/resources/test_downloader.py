import tempfile
import os
import contextlib
import pytest
from unittest import mock
from pytorchltr.dataset.resources.downloader import Downloader
from pytorchltr.dataset.resources.downloader import LoggingProgress
from pytorchltr.dataset.resources.downloader import TerminalProgress
from pytorchltr.dataset.resources.util import ChecksumError


@contextlib.contextmanager
def mock_urlopen(read, info):
    with mock.patch("pytorchltr.dataset.resources.downloader.urlopen") as urlopen_mock:
        urlopen_obj = mock.MagicMock()
        urlopen_obj.read.side_effect = read
        urlopen_obj.info.return_value = info
        urlopen_obj.__enter__.return_value = urlopen_obj
        urlopen_obj.__exit__.return_value = None
        urlopen_mock.return_value = urlopen_obj
        yield


def test_basic_download():
    with mock_urlopen(read=[b"mocked", b"content"], info={}):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = Downloader("http://mocked", "file.dat")
            downloader.download(tmpdir)
            with open(os.path.join(tmpdir, "file.dat"), "rb") as f:
                assert f.read() == b"mockedcontent"


def test_empty_download():
    with mock_urlopen(read=[], info={}):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = Downloader("http://mocked", "file.dat")
            downloader.download(tmpdir)
            with open(os.path.join(tmpdir, "file.dat"), "rb") as f:
                assert f.read() == b""


def test_download_sha256_succeeds():
    with mock_urlopen(read=[b"mocked", b"content"], info={}):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = Downloader(
                "http://mocked", "file.dat",
                sha256_checksum="d6185849aea1f15d847aba8b45cf54da8f9bc5895484d2f04830941f68148864")
            downloader.download(tmpdir)


def test_download_sha256_fails():
    with mock_urlopen(read=[b"mocked", b"content"], info={}):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = Downloader(
                "http://mocked", "file.dat",
                sha256_checksum="f13b7a15b26814ac2e5c8ab37ef135623ee0aed1cac5cc583e1b99fa688e0e29")
            with pytest.raises(ChecksumError) as excinfo:
                downloader.download(tmpdir)


def test_download_only_once():
    with mock_urlopen(read=[b"mocked", b"content"], info={}):
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_fn = mock.MagicMock()
            downloader = Downloader("http://mocked", "file.dat",
                                    progress_fn=progress_fn)

            # Should trigger download and 4 progress updates
            assert progress_fn.call_count == 0
            downloader.download(tmpdir)
            assert progress_fn.call_count == 4

            # Should NOT trigger download and no more progress updates
            downloader.download(tmpdir)
            assert progress_fn.call_count == 4


def test_download_postprocess_fn():
    with mock_urlopen(read=[b"mocked", b"content"], info={}):
        with tempfile.TemporaryDirectory() as tmpdir:
            postprocess_fn = mock.MagicMock()
            downloader = Downloader("http://mocked", "file.dat",
                                    postprocess_fn=postprocess_fn)
            downloader.download(tmpdir)
            postprocess_fn.assert_has_calls([
                mock.call(os.path.join(tmpdir, "file.dat"), tmpdir)
            ])


def test_download_progress_with_known_length():
    with mock_urlopen(read=[b"mocked", b"content"],
                      info={"Content-Length": "13"}):
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_fn = mock.MagicMock()
            downloader = Downloader("http://mocked", "file.dat",
                                    progress_fn=progress_fn)
            downloader.download(tmpdir)
            progress_fn.assert_has_calls([
                mock.call(0, 13, False),  # Initial progress (0 bytes)
                mock.call(6, 13, False),  # Should have read 6 bytes b"mocked"
                mock.call(13, 13, False), # Should have read 7 bytes b"content"
                mock.call(13, 13, True)   # Final call should be True
            ])


def test_download_progress_with_unknown_length():
    with mock_urlopen(read=[b"moc", b"ked", b"con", b"ten", b"t"], info={}):
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_fn = mock.MagicMock()
            downloader = Downloader("http://mocked", "file.dat",
                                    progress_fn=progress_fn)
            downloader.download(tmpdir)
            progress_fn.assert_has_calls([
                mock.call(0, None, False),  # Initial progress (0 bytes)
                mock.call(3, None, False),  # Should have read 3 bytes b"moc"
                mock.call(6, None, False),  # Should have read 3 bytes b"ked"
                mock.call(9, None, False),  # Should have read 3 bytes b"con"
                mock.call(12, None, False), # Should have read 3 bytes b"ten"
                mock.call(13, None, False), # Should have read 1 byte b"t"
                mock.call(13, None, True)   # Final call should be True
            ])


def test_download_logging_progress_with_known_length():
    with mock.patch("pytorchltr.dataset.resources.downloader.logging") as logging_mock:
        with mock_urlopen(read=[b"mocked", b"content"],
                          info={"Content-Length": "13"}):
            with tempfile.TemporaryDirectory() as tmpdir:
                downloader = Downloader(
                    "http://mocked", "file.dat",
                    progress_fn=LoggingProgress(interval=0.0))
                downloader.download(tmpdir)
                logging_mock.info.assert_has_calls([
                    mock.call("downloading   0% [0B / 13B]"),
                    mock.call("downloading  46% [6B / 13B]"),
                    mock.call("downloading 100% [13B / 13B]"),
                    mock.call("finished downloading [13B]"),
                ])


def test_download_logging_progress_with_unknown_length():
    with mock.patch("pytorchltr.dataset.resources.downloader.logging") as logging_mock:
        with mock_urlopen(read=[b"mocked", b"content"], info={}):
            with tempfile.TemporaryDirectory() as tmpdir:
                downloader = Downloader(
                    "http://mocked", "file.dat",
                    progress_fn=LoggingProgress(interval=0.0))
                downloader.download(tmpdir)
                logging_mock.info.assert_has_calls([
                    mock.call("downloading [0B / ?]"),
                    mock.call("downloading [6B / ?]"),
                    mock.call("downloading [13B / ?]"),
                    mock.call("finished downloading [13B]"),
                ])


def test_download_logging_progress_with_kilobytes():
    with mock.patch("pytorchltr.dataset.resources.downloader.logging") as logging_mock:
        with mock_urlopen(read=[b"mocked" * 1024, b"content" * 1024],
                          info={"Content-Length": "13312"}):
            with tempfile.TemporaryDirectory() as tmpdir:
                downloader = Downloader(
                    "http://mocked", "file.dat",
                    progress_fn=LoggingProgress(interval=0.0))
                downloader.download(tmpdir)
                logging_mock.info.assert_has_calls([
                    mock.call("downloading   0% [0B / 13KB]"),
                    mock.call("downloading  46% [6.0KB / 13KB]"),
                    mock.call("downloading 100% [13KB / 13KB]"),
                    mock.call("finished downloading [13KB]"),
                ])


def test_download_logging_progress_with_megabytes():
    with mock.patch("pytorchltr.dataset.resources.downloader.logging") as logging_mock:
        with mock_urlopen(read=[b"moc" * 1024 * 120, b"moc" * 1024 * 748, b"ked" * 1024 * 768, b"content123" * 1024 * 300],
                          info={"Content-Length": "8097792"}):
            with tempfile.TemporaryDirectory() as tmpdir:
                downloader = Downloader(
                    "http://mocked", "file.dat",
                    progress_fn=LoggingProgress(interval=0.0))
                downloader.download(tmpdir)
                logging_mock.info.assert_has_calls([
                    mock.call("downloading   0% [0B / 7.7MB]"),
                    mock.call("downloading   4% [360KB / 7.7MB]"),
                    mock.call("downloading  32% [2.5MB / 7.7MB]"),
                    mock.call("downloading  62% [4.8MB / 7.7MB]"),
                    mock.call("downloading 100% [7.7MB / 7.7MB]"),
                    mock.call("finished downloading [7.7MB]"),
                ])


def test_download_terminal_progress():
    with mock.patch("builtins.print") as print_mock:
        with mock_urlopen(read=[b"mocked", b"content"],
                          info={"Content-Length": "13"}):
            with tempfile.TemporaryDirectory() as tmpdir:
                downloader = Downloader(
                    "http://mocked", "file.dat",
                    progress_fn=TerminalProgress(interval=0.0))
                downloader.download(tmpdir)
                print_mock.assert_has_calls([
                    mock.call("\033[Kdownloading   0% [0B / 13B]", end="\r"),
                    mock.call("\033[Kdownloading  46% [6B / 13B]", end="\r"),
                    mock.call("\033[Kdownloading 100% [13B / 13B]", end="\r"),
                    mock.call("\033[Kfinished downloading [13B]", end="\n"),
                ])
