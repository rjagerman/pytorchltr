import os
import tempfile
from unittest import mock
from pathlib import Path

import pytest
from pytorchltr.datasets.util.file import validate_expected_files
from pytorchltr.datasets.util.file import validate_and_download
from pytorchltr.datasets.util.file import ChecksumError


def test_validate_expected_files_simple_succeeds():
    expected_files = ["file1.txt", "file2.txt"]
    actual_files = ["file1.txt", "file2.txt"]

    with tempfile.TemporaryDirectory() as tmpdir:

        # Create actual files
        for actual_file in actual_files:
            Path(os.path.join(tmpdir, actual_file)).touch()

        # Assert that validation succeeds
        validate_expected_files(tmpdir, expected_files)


def test_validate_expected_files_simple_fails():
    expected_files = ["file1.txt", "file2.txt", "file3.txt"]
    actual_files = ["file1.txt", "file2.txt"]

    with tempfile.TemporaryDirectory() as tmpdir:

        # Create actual files
        for actual_file in actual_files:
            Path(os.path.join(tmpdir, actual_file)).touch()

        # Assert that validation raises file not found error
        with pytest.raises(FileNotFoundError):
            validate_expected_files(tmpdir, expected_files)


def test_validate_expected_files_checksum_succeeds():
    expected_files = [
        {"path": "file1.txt", "sha256": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"},  # noqa: E501
        {"path": "file2.txt", "sha256": "486ea46224d1bb4fb680f34f7c9ad96a8f24ec88be73ea8e5a6c65260e9cb8a7"}  # noqa: E501
    ]
    actual_files = [
        {"path": "file1.txt", "contents": "hello"},
        {"path": "file2.txt", "contents": "world"}
    ]

    with tempfile.TemporaryDirectory() as tmpdir:

        # Create actual files
        for actual_file in actual_files:
            path = os.path.join(tmpdir, actual_file["path"])
            with open(path, "wt") as file_handle:
                file_handle.write(actual_file["contents"])

        # Assert that validation succeeds
        validate_expected_files(
            tmpdir, expected_files, validate_checksums=True)


def test_validate_expected_files_checksum_fails():
    expected_files = [
        {"path": "file1.txt", "sha256": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"},  # noqa: E501
        {"path": "file2.txt", "sha256": "486ea46224d1bb4fb680f34f7c9ad96a8f24ec88be73ea8e5a6c65260e9cb8a7"}  # noqa: E501
    ]
    actual_files = [
        {"path": "file1.txt", "contents": "goodbye"},
        {"path": "file2.txt", "contents": "world"}
    ]

    with tempfile.TemporaryDirectory() as tmpdir:

        # Create actual files
        for actual_file in actual_files:
            path = os.path.join(tmpdir, actual_file["path"])
            with open(path, "wt") as file_handle:
                file_handle.write(actual_file["contents"])

        # Assert that validation fails with checksum error
        with pytest.raises(ChecksumError):
            validate_expected_files(
                tmpdir, expected_files, validate_checksums=True)


def test_validate_expected_files_format_fails():
    expected_files = [
        {"filepath": "file1.txt", "sha256": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"},  # noqa: E501
        {"path": "file2.txt", "checksum": "486ea46224d1bb4fb680f34f7c9ad96a8f24ec88be73ea8e5a6c65260e9cb8a7"}  # noqa: E501
    ]
    actual_files = [
        {"path": "file1.txt", "contents": "hello"},
        {"path": "file2.txt", "contents": "world"}
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create actual files
        for actual_file in actual_files:
            path = os.path.join(tmpdir, actual_file["path"])
            with open(path, "wt") as file_handle:
                file_handle.write(actual_file["contents"])

        # Assert that validation fails with checksum error
        with pytest.raises(ValueError):
            validate_expected_files(
                tmpdir, expected_files, validate_checksums=True)


def test_validate_and_download_calls_download():
    downloader = mock.MagicMock()
    expected_files = ["file1.txt"]
    actual_files = ["file1.txt"]

    def create_files_side_effect(location):
        for actual_file in actual_files:
            Path(os.path.join(location, actual_file)).touch()

    downloader.download.side_effect = create_files_side_effect

    with tempfile.TemporaryDirectory() as tmpdir:
        # Call validate_and_download and assert download gets triggered.
        validate_and_download(
            tmpdir, expected_files, downloader=downloader,
            validate_checksums=False)
        downloader.download.assert_called_once_with(tmpdir)


def test_validate_and_download_skips_download():
    downloader = mock.MagicMock()
    expected_files = ["file1.txt"]
    actual_files = ["file1.txt"]

    def create_files_side_effect(location):
        for actual_file in actual_files:
            Path(os.path.join(location, actual_file)).touch()

    downloader.download.side_effect = create_files_side_effect

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files already, so that validate_and_download can skip the
        # download call.
        create_files_side_effect(tmpdir)

        # Call validate_and_download and assert download was not triggered.
        validate_and_download(
            tmpdir, expected_files, downloader=downloader,
            validate_checksums=False)
        downloader.download.assert_not_called()


def test_validate_and_download_fails_after_download_fails():
    downloader = mock.MagicMock()
    expected_files = ["file1.txt"]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Call validate_and_download and assert that file not found errors
        # is raised when no files are actually downloaded.
        with pytest.raises(FileNotFoundError):
            validate_and_download(
                tmpdir, expected_files, downloader=downloader,
                validate_checksums=False)

            # Assert download was called
            downloader.download.assert_called_once_with(tmpdir)


def test_validate_and_download_fails_without_downloader():
    expected_files = ["file1.txt"]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Call validate_and_download and assert that file not found error
        # is raised when no files are actually downloaded.
        with pytest.raises(FileNotFoundError):
            validate_and_download(
                tmpdir, expected_files, downloader=None,
                validate_checksums=False)


def test_validate_and_download_succeeds_without_downloader():
    expected_files = ["file1.txt"]
    actual_files = ["file1.txt"]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create actual files.
        for actual_file in actual_files:
            Path(os.path.join(tmpdir, actual_file)).touch()

        # Call validate_and_download and make sure nothing gets raised.
        validate_and_download(
            tmpdir, expected_files, downloader=None,
            validate_checksums=False)
